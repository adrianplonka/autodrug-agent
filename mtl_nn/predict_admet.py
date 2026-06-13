"""Predict ADMET properties for a CSV of molecules using the trained MTL net.

Loads the hard-parameter-sharing ADMET network and its weights, featurizes
every SMILES in an input CSV to a Morgan fingerprint, runs a single batched
forward pass, and writes the input rows back out with one prediction column
per task.

Classification heads (bbb, cyp3a4, herg) emit probabilities in [0, 1] -> the
columns are suffixed ``_prob``. Regression heads emit the raw target; the only
inverted transform is ``half_life`` (trained on log1p, so we apply expm1 to get
hours back), matching the training pipeline.

Run from the ``autodrug-agent`` directory:

    python -m mtl_nn.predict_admet
    python -m mtl_nn.predict_admet --input path/to/molecules.csv --output preds.csv
"""
from __future__ import annotations

# Must be set before torch / RDKit pull in their OpenMP runtimes, otherwise
# the two libiomp copies abort with "OMP: Error #15" on Windows.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import FP_NBITS, FP_RADIUS, TASKS, resolve_device
from .data.featurizer import batch_smiles_to_fps
from .model import ADMET_MTL_Network

# Repo root (…/leki_projekt) holds both the weights and the agent's CSV output.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WEIGHTS = _REPO_ROOT / "admet_mtl_weights.pt"
DEFAULT_INPUT = _REPO_ROOT / "novel_molecules.csv"
DEFAULT_OUTPUT = _REPO_ROOT / "novel_molecules_admet.csv"

# Per-task one-liners purely for the console legend (not used for math).
TASK_DESCRIPTIONS: dict[str, str] = {
    "caco2":      "Caco-2 permeability (log Papp)",
    "bbb":        "blood-brain-barrier penetration (probability)",
    "cyp3a4":     "CYP3A4 inhibition (probability)",
    "half_life":  "half-life in hours (expm1 of log1p target)",
    "herg":       "hERG block (probability)",
    "hum_mic_cl": "human microsomal clearance",
    "ppb":        "plasma protein binding",
    "logd":       "logD (lipophilicity)",
    "water_sol":  "aqueous solubility (logS)",
}


def column_name(spec) -> str:
    """Prediction column name: probabilities get a ``_prob`` suffix."""
    return f"{spec.name}_prob" if spec.task_type == "classification" else spec.name


def load_input_csv(path: Path) -> pd.DataFrame:
    """Read the molecules CSV, tolerating an inconsistent column count.

    The agent's ``novel_molecules.csv`` can end up ragged: rows written before
    and after the property columns were added have different widths while the
    on-disk header keeps the original schema. A plain ``pd.read_csv`` aborts on
    that, so we fall back to a positional reader that aligns every row to the
    header (extra trailing fields are dropped, short rows are padded).
    """
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        print(f"[predict] {path.name} has an inconsistent column count - "
              "reading defensively (aligning rows to the header).")
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            raise ValueError(f"{path} is empty")
        header = rows[0]
        width = len(header)
        fixed = [(r + [""] * width)[:width] for r in rows[1:]]
        return pd.DataFrame(fixed, columns=header)


def load_model(weights_path: Path, device: torch.device) -> ADMET_MTL_Network:
    """Build the network and load the trained weights in eval mode."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    model = ADMET_MTL_Network(TASKS)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_domain(
    smiles: list[str],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Applicability-domain check via max Tanimoto similarity to the train set.

    Rebuilds the exact training split (same TDC/PharmaBench merge and seed as
    training), fingerprints it, and for every input molecule reports the
    maximum Tanimoto similarity to any training molecule. Predictions for
    molecules below ``threshold`` are extrapolations and should be treated with
    caution.

    Requires the training stack (PyTDC) and network access, so it is imported
    lazily and only when the domain check is requested.

    Returns ``(tanimoto_max, in_domain)`` arrays aligned with ``smiles``;
    unparseable SMILES get NaN / False.
    """
    from .config import TASKS  # local: keep module import light
    from .data import build_dataloaders, download_and_merge_tdc
    from .data.featurizer import extract_train_fps, tanimoto_max_similarity

    print("[domain] downloading + merging TDC/PharmaBench training data ...")
    df = download_and_merge_tdc(TASKS)
    # 4th return value is the list of training-split SMILES (same seed as fit).
    *_, train_smiles = build_dataloaders(df, TASKS)
    train_fps = extract_train_fps(train_smiles)

    print(f"[domain] scoring {len(smiles)} molecules "
          f"(Tanimoto threshold = {threshold}) ...")
    sims = np.array(
        [tanimoto_max_similarity(s, train_fps) for s in smiles],
        dtype=object,
    )
    sims = np.array([np.nan if v is None else float(v) for v in sims],
                    dtype=np.float64)
    in_domain = sims >= threshold
    n_in = int(np.nansum(in_domain))
    print(f"[domain] in-domain (>= {threshold}): {n_in}/{len(smiles)} "
          f"({n_in / len(smiles) * 100:.0f}%)")
    return sims, in_domain


@torch.no_grad()
def predict_frame(
    model: ADMET_MTL_Network,
    smiles: list[str],
    device: torch.device,
    batch_size: int = 512,
    mc_samples: int = 10,
) -> dict[str, np.ndarray]:
    """Predict every task for a list of SMILES.

    With ``mc_samples > 1`` (the default), each task gets a Monte-Carlo-Dropout
    mean column plus a ``<col>_std`` uncertainty column. With ``mc_samples <= 1``
    a single deterministic pass is used (dropout off) and no std columns are
    produced.

    Returns a dict {column_name: array aligned to ``smiles``}. Rows whose SMILES
    RDKit cannot parse are filled with NaN so the output stays aligned with the
    input CSV.
    """
    use_mc = mc_samples and mc_samples > 1
    fps, valid_idx = batch_smiles_to_fps(smiles, radius=FP_RADIUS, n_bits=FP_NBITS)
    n = len(smiles)

    # Pre-build all output columns as NaN so unparseable rows stay aligned.
    columns: dict[str, np.ndarray] = {}
    for spec in model.tasks:
        columns[column_name(spec)] = np.full(n, np.nan, dtype=np.float64)
        if use_mc:
            columns[f"{column_name(spec)}_std"] = np.full(n, np.nan, dtype=np.float64)

    if fps.shape[0] == 0:
        return columns

    # Accumulate per-task mean (and std) for the valid rows, batch by batch.
    means: dict[str, list[float]] = {spec.name: [] for spec in model.tasks}
    stds: dict[str, list[float]] = {spec.name: [] for spec in model.tasks}
    for start in range(0, fps.shape[0], batch_size):
        chunk = torch.from_numpy(fps[start:start + batch_size]).to(device)
        if use_mc:
            out = model.predict_mc_dropout(chunk, n_samples=mc_samples)
            for spec in model.tasks:
                mean_t, std_t = out[spec.name]
                means[spec.name].extend(mean_t.numpy().tolist())
                stds[spec.name].extend(std_t.numpy().tolist())
        else:
            out = model.predict(chunk)  # sigmoid already applied to classification
            for spec in model.tasks:
                means[spec.name].extend(out[spec.name].cpu().numpy().tolist())

    for spec in model.tasks:
        mean = np.asarray(means[spec.name], dtype=np.float64)
        std = np.asarray(stds[spec.name], dtype=np.float64) if use_mc else None
        # Invert the log1p transform applied to half_life during training.
        # Do it before aggregating into hours: the std band is mapped through
        # expm1 symmetrically so the reported uncertainty is in hours too.
        if spec.name == "half_life":
            if use_mc:
                half_band = (np.expm1(mean + std) - np.expm1(mean - std)) / 2.0
                std = half_band
            mean = np.expm1(mean)
        columns[column_name(spec)][valid_idx] = mean
        if use_mc:
            columns[f"{column_name(spec)}_std"][valid_idx] = std

    n_failed = n - len(valid_idx)
    if n_failed:
        print(f"[predict] {n_failed} SMILES could not be parsed (left as NaN)")
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"input CSV with a 'smiles' column (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output CSV path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help=f"model weights .pt (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--smiles-col", default="smiles",
                        help="name of the SMILES column (default: smiles)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--mc-samples", type=int, default=10,
                        help="Monte-Carlo-Dropout passes for uncertainty "
                             "(default: 10; use 1 for a single deterministic pass)")
    parser.add_argument("--domain-check", action="store_true",
                        help="add Tanimoto applicability-domain columns "
                             "(needs PyTDC + internet; run in the mldd26 env)")
    parser.add_argument("--tanimoto-threshold", type=float, default=0.4,
                        help="max-Tanimoto cutoff for in-domain (default: 0.4)")
    args = parser.parse_args()

    device = resolve_device()
    print(f"[predict] device: {device}")

    df = load_input_csv(args.input)
    if args.smiles_col not in df.columns:
        raise KeyError(
            f"Column '{args.smiles_col}' not in {args.input}. "
            f"Available columns: {list(df.columns)}"
        )
    smiles = df[args.smiles_col].astype(str).tolist()
    print(f"[predict] loaded {len(smiles)} molecules from {args.input}")

    model = load_model(args.weights, device)
    print(f"[predict] loaded weights from {args.weights}")
    if args.mc_samples and args.mc_samples > 1:
        print(f"[predict] MC-Dropout uncertainty: {args.mc_samples} passes "
              "(adds a _std column per task)")
    else:
        print("[predict] deterministic single pass (no uncertainty)")
    print("[predict] tasks:")
    for spec in model.tasks:
        kind = "prob" if spec.task_type == "classification" else "value"
        print(f"  - {column_name(spec):<16s} [{kind}] {TASK_DESCRIPTIONS.get(spec.name, '')}")

    preds = predict_frame(model, smiles, device,
                          batch_size=args.batch_size, mc_samples=args.mc_samples)

    out_df = df.copy()
    for col, values in preds.items():
        out_df[col] = np.round(values, 4)

    if args.domain_check:
        tanimoto_max, in_domain = compute_domain(smiles, args.tanimoto_threshold)
        out_df["tanimoto_max"] = np.round(tanimoto_max, 4)
        out_df["in_domain"] = in_domain

    out_df.to_csv(args.output, index=False)
    n_extra = len(preds) + (2 if args.domain_check else 0)
    print(f"\n[predict] wrote {len(out_df)} rows x {n_extra} prediction columns to {args.output}")

    # Quick sanity summary so the user sees the ranges at a glance.
    pred_cols = list(preds.keys())
    summary = out_df[pred_cols].describe().loc[["mean", "min", "max"]].T
    print("\n[predict] prediction summary (mean / min / max):")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
