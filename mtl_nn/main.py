"""Entry point: download data, build the MTL model, train, and report.

Run from the parent directory:

    python -m mtl_nn.main
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score
import pandas as pd
from pathlib import Path

from .config import EPOCHS, RANDOM_SEED, TASKS, configure_cuda_runtime, resolve_device
from .data import build_dataloaders, download_and_merge_tdc
from .model import ADMET_MTL_Network
from .training import Trainer
from .data.featurizer import extract_train_fps, tanimoto_max_similarity

TANIMOTO_THRESHOLD = 0.4
MC_SAMPLES = 50

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: ADMET_MTL_Network,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """Compute per-task metrics on a loader, respecting the label mask."""
    model.eval()
    # Collect predictions and targets per task, dropping rows that lack a label.
    bucket: dict[str, list[tuple[float, float]]] = {t.name: [] for t in model.tasks}
    for fps, targets, mask in loader:
        fps = fps.to(device)
        preds = model.predict(fps)  # already sigmoid'd for classification heads
        for k, spec in enumerate(model.tasks):
            m = mask[:, k].bool()
            if not m.any():
                continue
            y_pred = preds[spec.name].detach().cpu()[m].numpy()
            y_true = targets[:, k][m].numpy()
            bucket[spec.name].extend(zip(y_pred.tolist(), y_true.tolist()))

    print("\n[test] per-task metrics:")
    for spec in model.tasks:
        if not bucket[spec.name]:
            print(f"  - {spec.name}: no labels in this split")
            continue
        y_pred, y_true = map(np.array, zip(*bucket[spec.name]))
        if spec.task_type == "classification":
            try:
                metric = roc_auc_score(y_true, y_pred)
                print(f"  - {spec.name:<10s} ROC-AUC = {metric:.4f}  (n={len(y_true)})")
            except ValueError:
                # Single-class split — AUC is undefined.
                print(f"  - {spec.name:<10s} ROC-AUC = n/a (single class)  (n={len(y_true)})")
        else:
            if spec.name == "half_life":
                y_pred = np.expm1(y_pred)
                y_true = np.expm1(y_true)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            print(f"  - {spec.name:<10s} RMSE    = {rmse:.4f}  (n={len(y_true)})")



def predict_with_domain_check(
    model: ADMET_MTL_Network,
    smiles_list: list[str],
    train_fps: list,
    device: torch.device,
    threshold: float = TANIMOTO_THRESHOLD,
    n_mc_samples: int = MC_SAMPLES,
) -> None:
    print(f"\n[predict] predykcja dla {len(smiles_list)} cząsteczek")
    print(f"[predict] próg Tanimoto: {threshold} | MC samples: {n_mc_samples}\n")

    for smi in smiles_list:
        # Tanimoto check
        max_sim = tanimoto_max_similarity(smi, train_fps)
        if max_sim is None:
            print(f"  {smi}: nieprawidłowy SMILES\n")
            continue

        in_domain = max_sim >= threshold
        domain_str = "✓ w domenie" if in_domain else "⚠ POZA DOMENĄ"

        # Fingerprint → tensor
        from .data.featurizer import smiles_to_morgan_fp
        fp_arr = smiles_to_morgan_fp(smi)
        if fp_arr is None:
            continue
        fp_tensor = torch.tensor(fp_arr).unsqueeze(0).to(device)

        # MC Dropout predykcja
        mc_results = model.predict_mc_dropout(fp_tensor, n_samples=n_mc_samples)

        # Wypisz
        print(f"  SMILES: {smi}")
        print(f"  Tanimoto max: {max_sim:.3f}  {domain_str}")
        print(f"  {'Task':<12s}  {'Mean':>8s}  {'±Std':>8s}  Pewność")
        print(f"  {'-'*48}")
        for spec in model.tasks:
            mean, std = mc_results[spec.name]
            if spec.name == "half_life":
                mean = float(np.expm1(mean))
            if std < 0.05:
                confidence = "wysoka"
            elif std < 0.15:
                confidence = "srednia"
            else:
                confidence = "niska"
            unit = "P=" if spec.task_type == "classification" else "  "
            print(f"  {spec.name:<12s}: {unit}{mean:>7.4f}  ±{std:.4f}  {confidence}")
        print()



def main() -> None:
    _set_seed(RANDOM_SEED)
    configure_cuda_runtime()
    device = resolve_device()
    print(f"[main] resolved device: {device}")

    df = download_and_merge_tdc(TASKS)
    train_loader, val_loader, test_loader, train_smiles = build_dataloaders(df, TASKS)

    model = ADMET_MTL_Network(TASKS)
    print(model)

    trainer = Trainer(model, TASKS, device=device)
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    evaluate(model, test_loader, trainer.device)

    # Tanimoto — pobierz SMILES treningowe z loadera
    train_smiles = df.iloc[train_loader.dataset.indices]["smiles"].tolist()
    train_fps = extract_train_fps(train_smiles)

    # Predykcja na nowych cząsteczkach
    novel_path = Path(__file__).parent.parent / "agent_base_model" / "novel_molecules.csv"
    novel_df = pd.read_csv(novel_path)
    new_smiles = novel_df["smiles"].tolist()

    print(f"[predict] wczytano {len(new_smiles)} nowych cząsteczek")
    predict_with_domain_check(model, new_smiles, train_fps, device)


if __name__ == "__main__":
    main()
