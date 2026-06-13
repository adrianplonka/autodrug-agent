"""Download TDC ADMET datasets and PharmaBench datasets, then merge on SMILES.

The merged DataFrame has one row per unique molecule and one column per task.
Most molecules will only have a label for one (or a few) of the tasks; the
remaining cells are ``NaN`` and are masked out by the loss function.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from tdc.single_pred import ADME, Tox

from ..config import TaskSpec

_GROUP_TO_CLASS = {"ADME": ADME, "Tox": Tox}

_PHARMA_BASE_URL = (
    "https://raw.githubusercontent.com/mindrank-ai/PharmaBench/main/data/final_datasets/"
)
_PHARMA_FILES: dict[str, str] = {
    "hum_mic_cl": "hum_mic_cl_reg_final_data.csv",
    "ppb":        "ppb_reg_final_data.csv",
    "logd":       "logd_reg_final_data.csv",
    "water_sol":  "water_sol_reg_final_data.csv",
}


def _canon(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(mol) if mol else None


def _fetch_tdc(spec: TaskSpec) -> pd.DataFrame:
    cls = _GROUP_TO_CLASS[spec.tdc_group]
    df = cls(name=spec.tdc_name).get_data()
    out = df[["Drug", "Y"]].rename(columns={"Drug": "smiles", "Y": spec.name})
    out = out.groupby("smiles", as_index=False)[spec.name].mean()
    if spec.task_type == "classification":
        out = out[out[spec.name].isin([0.0, 1.0])]
    if spec.name == "half_life":
        out[spec.name] = np.log1p(out[spec.name])
    return out


def _fetch_pharma(task_name: str) -> pd.DataFrame:
    fname = _PHARMA_FILES[task_name]
    df = pd.read_csv(f"{_PHARMA_BASE_URL}{fname}")
    df = df[["Smiles_unify", "value"]].rename(
        columns={"Smiles_unify": "smiles", "value": task_name}
    )
    df["smiles"] = df["smiles"].apply(_canon)
    df = df.dropna(subset=["smiles", task_name])
    df = df.groupby("smiles", as_index=False)[task_name].mean()
    return df


def download_and_merge_tdc(tasks: Sequence[TaskSpec]) -> pd.DataFrame:
    """Download all datasets and outer-merge them on SMILES.

    TDC tasks use the PyTDC API; tasks without ``tdc_name`` are fetched from
    PharmaBench if their name appears in ``_PHARMA_FILES``.

    Returns a DataFrame with columns ``['smiles', task1, task2, ...]``.
    Missing labels appear as ``NaN`` — that is the signal the masked loss
    uses to skip a sample for a given task.
    """
    merged: pd.DataFrame | None = None

    for spec in tasks:
        if spec.tdc_name is not None:
            print(f"[tdc] downloading {spec.tdc_name} ...")
            part = _fetch_tdc(spec)
        elif spec.name in _PHARMA_FILES:
            print(f"[pharma] downloading {spec.name} from PharmaBench ...")
            part = _fetch_pharma(spec.name)
        else:
            print(f"[skip] {spec.name}: no data source configured")
            continue

        merged = part if merged is None else merged.merge(part, on="smiles", how="outer")

    assert merged is not None, "TASKS list was empty or all tasks skipped"

    print(
        f"\n[data] merged dataset: {len(merged)} unique molecules, "
        f"{len(tasks)} tasks. Per-task label coverage:"
    )
    for spec in tasks:
        n = merged[spec.name].notna().sum()
        print(f"  - {spec.name:<10s}: {n:>6d} labeled ({n / len(merged):.1%})")
    return merged
