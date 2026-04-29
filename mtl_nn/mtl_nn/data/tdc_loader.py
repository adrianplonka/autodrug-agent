"""Download multiple TDC ADMET datasets and merge them on SMILES.

The merged DataFrame has one row per unique molecule and one column per task.
Most molecules will only have a label for one (or a few) of the tasks; the
remaining cells are ``NaN`` and are masked out by the loss function.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
from tdc.single_pred import ADME, Tox

from ..config import TaskSpec


_GROUP_TO_CLASS = {"ADME": ADME, "Tox": Tox}


def _fetch_one(spec: TaskSpec) -> pd.DataFrame:
    """Pull a single TDC dataset and return a 2-column DataFrame [SMILES, <task>]."""
    cls = _GROUP_TO_CLASS[spec.tdc_group]
    df = cls(name=spec.tdc_name).get_data()  # columns: 'Drug_ID', 'Drug', 'Y'
    out = df[["Drug", "Y"]].rename(columns={"Drug": "smiles", "Y": spec.name})
    # If the same SMILES appears multiple times (rare but possible), average
    # regression labels and majority-vote-via-mean classification labels.
    out = out.groupby("smiles", as_index=False)[spec.name].mean()
    return out


def download_and_merge_tdc(tasks: Sequence[TaskSpec]) -> pd.DataFrame:
    """Download every dataset in ``tasks`` and outer-merge them on SMILES.

    Returns a DataFrame with columns ``['smiles', task1, task2, ...]``.
    Missing labels appear as ``NaN`` — that is the signal the masked loss
    uses to skip a sample for a given task.
    """
    merged: pd.DataFrame | None = None
    for spec in tasks:
        print(f"[tdc] downloading {spec.tdc_name} ({spec.task_type}) ...")
        part = _fetch_one(spec)
        merged = part if merged is None else merged.merge(part, on="smiles", how="outer")

    assert merged is not None, "TASKS list was empty"
    print(
        f"[tdc] merged dataset: {len(merged)} unique molecules, "
        f"{len(tasks)} tasks. Per-task label coverage:"
    )
    for spec in tasks:
        n = merged[spec.name].notna().sum()
        print(f"  - {spec.name:<10s}: {n:>6d} labeled ({n / len(merged):.1%})")
    return merged
