"""PyTorch Dataset and DataLoader factory for the merged ADMET corpus.

Each item yields:
    fingerprint : FloatTensor (n_bits,)
    targets     : FloatTensor (n_tasks,)   -- NaNs preserved
    mask        : FloatTensor (n_tasks,)   -- 1.0 where the label exists, 0.0 otherwise

The mask is precomputed here (rather than at loss time) so the training
loop is dead-simple and the masking semantics live next to the data.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ..config import (
    BATCH_SIZE,
    FP_NBITS,
    FP_RADIUS,
    RANDOM_SEED,
    TEST_FRACTION,
    VAL_FRACTION,
    TaskSpec,
)
from .featurizer import batch_smiles_to_fps


class ADMETMultiTaskDataset(Dataset):
    """In-memory dataset of (fingerprint, target_vector, mask_vector) triples."""

    def __init__(
        self,
        fingerprints: np.ndarray,   # (N, n_bits) float32
        targets: np.ndarray,        # (N, n_tasks) float32, NaN for missing
        mask: np.ndarray,           # (N, n_tasks) float32, 1 where labeled
    ) -> None:
        assert fingerprints.shape[0] == targets.shape[0] == mask.shape[0]
        # Replace NaNs in targets with 0 *after* the mask is built; the loss
        # multiplies by the mask so the placeholder value is never read,
        # but keeping NaNs in tensors triggers gradient-pollution bugs.
        targets = np.nan_to_num(targets, nan=0.0).astype(np.float32)
        self.fingerprints = torch.from_numpy(fingerprints.astype(np.float32))
        self.targets = torch.from_numpy(targets)
        self.mask = torch.from_numpy(mask.astype(np.float32))

    def __len__(self) -> int:
        return self.fingerprints.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.fingerprints[idx], self.targets[idx], self.mask[idx]


def _featurize_frame(
    df: pd.DataFrame,
    tasks: Sequence[TaskSpec],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run RDKit featurization and assemble target + mask matrices."""
    smiles = df["smiles"].tolist()
    fps, valid_idx = batch_smiles_to_fps(smiles, radius=FP_RADIUS, n_bits=FP_NBITS)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    target_cols = [spec.name for spec in tasks]
    targets = df_valid[target_cols].to_numpy(dtype=np.float32)   # NaN where missing
    mask = (~np.isnan(targets)).astype(np.float32)               # 1 where labeled
    return fps, targets, mask


def build_dataloaders(
    df: pd.DataFrame,
    tasks: Sequence[TaskSpec],
    batch_size: int = BATCH_SIZE,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Featurize, split (train / val / test), and wrap in DataLoaders."""
    fps, targets, mask = _featurize_frame(df, tasks)
    print(f"[dataset] featurized {fps.shape[0]} / {len(df)} molecules (rest unparseable)")

    # Two-stage split keeps the implementation independent of sklearn version
    # quirks around three-way splits.
    idx = np.arange(fps.shape[0])
    idx_train, idx_temp = train_test_split(
        idx, test_size=val_fraction + test_fraction, random_state=seed
    )
    rel_test = test_fraction / (val_fraction + test_fraction)
    idx_val, idx_test = train_test_split(idx_temp, test_size=rel_test, random_state=seed)

    def make(indices: np.ndarray, shuffle: bool) -> DataLoader:
        ds = ADMETMultiTaskDataset(fps[indices], targets[indices], mask[indices])
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return make(idx_train, True), make(idx_val, False), make(idx_test, False)
