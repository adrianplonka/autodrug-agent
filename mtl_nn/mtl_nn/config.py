"""Central configuration: task registry and hyperparameters.

The ``TASKS`` registry is the single source of truth for which TDC endpoints
the MTL model handles, what kind of problem each one is (classification vs.
regression), and which TDC sub-package they live in. The rest of the code
(data loading, model heads, loss function, metrics) iterates over this list
so adding/removing a task is a one-line change here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class TaskSpec:
    name: str             # Human-readable column name used in merged DataFrame
    tdc_name: str         # Exact dataset name in TDC
    tdc_group: str        # 'ADME' or 'Tox' — selects the tdc.single_pred class
    task_type: TaskType   # Drives loss function and output activation


# 5 ADMET endpoints — one per ADMET letter.
TASKS: Tuple[TaskSpec, ...] = (
    TaskSpec("caco2",     "Caco2_Wang",      "ADME", "regression"),     # A
    TaskSpec("bbb",       "BBB_Martins",     "ADME", "classification"), # D
    TaskSpec("cyp3a4",    "CYP3A4_Veith",    "ADME", "classification"), # M
    TaskSpec("half_life", "Half_Life_Obach", "ADME", "regression"),     # E
    TaskSpec("herg",      "hERG",            "Tox",  "classification"), # T
)


# --- Featurizer -------------------------------------------------------------
FP_RADIUS: int = 2
FP_NBITS: int = 2048

# --- Model ------------------------------------------------------------------
SHARED_HIDDEN_DIMS: Tuple[int, ...] = (1024, 512, 256)
HEAD_HIDDEN_DIM: int = 64
DROPOUT: float = 0.2

# --- Training ---------------------------------------------------------------
BATCH_SIZE: int = 256
EPOCHS: int = 30
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
VAL_FRACTION: float = 0.1
TEST_FRACTION: float = 0.1
RANDOM_SEED: int = 67
