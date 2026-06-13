"""SMILES -> Morgan fingerprint featurization."""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Silence RDKit's chatty C++ warnings about odd molecules — TDC contains some.
RDLogger.DisableLog("rdApp.*")


def smiles_to_morgan_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """Convert a single SMILES string to a Morgan fingerprint bit vector.

    Returns ``None`` if RDKit cannot parse the SMILES — callers are expected
    to drop those rows. Returning ``None`` rather than raising keeps the
    featurization vectorizable over noisy datasets.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    # ConvertToNumpyArray writes the bits directly into the preallocated buffer.
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def batch_smiles_to_fps(
    smiles_list: Iterable[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, List[int]]:
    """Featurize an iterable of SMILES strings.

    Returns
    -------
    fps : np.ndarray of shape (n_valid, n_bits)
        Stacked fingerprints for SMILES that parsed successfully.
    valid_indices : list[int]
        Original indices (into ``smiles_list``) that were kept. The caller
        uses this to align labels with fingerprints after dropping bad rows.
    """
    fps: List[np.ndarray] = []
    valid_indices: List[int] = []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_morgan_fp(smi, radius=radius, n_bits=n_bits)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)
    if not fps:
        return np.zeros((0, n_bits), dtype=np.float32), []
    return np.stack(fps, axis=0), valid_indices

def smiles_to_rdkit_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[Chem.DataStructs.ExplicitBitVect]:
    """Zwraca RDKit ExplicitBitVect (potrzebny do Tanimoto) lub None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def extract_train_fps(
    smiles_list: Iterable[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> List[Chem.DataStructs.ExplicitBitVect]:
    """Zbiera RDKit fingerprinity dla zbioru treningowego do Tanimoto check."""
    fps = []
    for smi in smiles_list:
        fp = smiles_to_rdkit_fp(smi, radius=radius, n_bits=n_bits)
        if fp is not None:
            fps.append(fp)
    print(f"[domain] zapisano {len(fps)} fingerprintów treningowych")
    return fps


def tanimoto_max_similarity(
    smiles: str,
    train_fps: List[Chem.DataStructs.ExplicitBitVect],
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[float]:
    """Liczy max Tanimoto similarity nowej cząsteczki do zbioru treningowego."""
    fp = smiles_to_rdkit_fp(smiles, radius=radius, n_bits=n_bits)
    if fp is None:
        return None
    similarities = Chem.DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    return float(max(similarities))