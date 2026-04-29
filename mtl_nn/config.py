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

import torch

TaskType = Literal["classification", "regression"]
DeviceMode = Literal["auto", "cuda", "cpu"]


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

# --- CUDA / device ----------------------------------------------------------
# DEVICE_MODE controls hardware selection:
#   "auto" — use CUDA if torch.cuda.is_available(), else fall back to CPU
#   "cuda" — force CUDA; raise if unavailable (catches silent CPU fallback in prod)
#   "cpu"  — force CPU (useful for debugging numeric issues without CUDA noise)
DEVICE_MODE: DeviceMode = "auto"

# Which GPU to use when multiple are present. Ignored on CPU.
CUDA_DEVICE_INDEX: int = 0

# torch.backends.cudnn.benchmark: lets cuDNN pick the fastest conv algorithm
# for the input shapes it sees. Safe to enable here because batch shape is
# fixed (Morgan FP length doesn't vary) — turn off if you ever feed variable
# input sizes, since the autotuner re-runs on every new shape.
CUDNN_BENCHMARK: bool = True

# Make cuDNN deterministic. Slower; only enable when you need bit-exact reruns.
CUDNN_DETERMINISTIC: bool = False

# Use TF32 matmul on Ampere+ GPUs (A100, RTX 30xx/40xx). ~2x speedup on linear
# layers with negligible accuracy impact for this problem.
ALLOW_TF32: bool = True

# Mixed-precision (autocast + GradScaler) for the training loop. Big speedup
# on Ampere+; no-op on CPU, where it is silently disabled.
USE_AMP: bool = True

# pin_memory in DataLoaders speeds up host->device transfers when on CUDA.
# Auto-resolved from the chosen device; exposed here so it can be overridden.
PIN_MEMORY: bool | None = None  # None => auto (True on CUDA, False on CPU)


def resolve_device() -> torch.device:
    """Resolve ``DEVICE_MODE`` into a concrete ``torch.device``.

    Centralized so every component (trainer, dataloaders, evaluation) agrees
    on which device to use, and so "force cuda but it's missing" fails loudly
    instead of silently training on CPU for hours.
    """
    if DEVICE_MODE == "cpu":
        return torch.device("cpu")
    if DEVICE_MODE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE_MODE='cuda' but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or set DEVICE_MODE='auto'."
            )
        return torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
    # auto
    if torch.cuda.is_available():
        return torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
    return torch.device("cpu")


def configure_cuda_runtime() -> None:
    """Apply global CUDA/cuDNN flags. Safe to call on CPU-only machines."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    torch.backends.cudnn.deterministic = CUDNN_DETERMINISTIC
    # TF32 toggles live on two separate backends in modern PyTorch.
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32


def use_pin_memory(device: torch.device) -> bool:
    """Resolve ``PIN_MEMORY`` (auto or explicit) against the active device."""
    if PIN_MEMORY is not None:
        return PIN_MEMORY
    return device.type == "cuda"
