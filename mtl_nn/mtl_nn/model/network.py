"""Hard-parameter-sharing MTL network for ADMET prediction.

Architecture:

    Morgan FP (n_bits)
          |
    [shared MLP: Linear -> BN -> ReLU -> Dropout]  x len(SHARED_HIDDEN_DIMS)
          |
    shared embedding (last hidden dim)
        /   |   |   |   \\
       H1  H2  H3  H4   H5         <- one head per task
       |   |   |   |    |
       y1  y2  y3  y4   y5         <- raw logits / regression outputs

The heads emit *raw* logits for classification tasks. The Sigmoid is applied
during inference (and inside ``BCEWithLogitsLoss`` during training) for
numerical stability, but ``predict`` returns calibrated probabilities so the
public interface still satisfies the spec ("Sigmoid for binary classification").
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from ..config import (
    DROPOUT,
    FP_NBITS,
    HEAD_HIDDEN_DIM,
    SHARED_HIDDEN_DIMS,
    TaskSpec,
)


def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Linear -> BatchNorm -> ReLU -> Dropout."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )


class _TaskHead(nn.Module):
    """Small per-task MLP that produces a single scalar output."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        return self.net(shared_repr).squeeze(-1)  # (B,)


class ADMET_MTL_Network(nn.Module):
    """Hard-parameter-sharing MTL net: one shared trunk, N task-specific heads."""

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        input_dim: int = FP_NBITS,
        shared_hidden_dims: Sequence[int] = SHARED_HIDDEN_DIMS,
        head_hidden_dim: int = HEAD_HIDDEN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.tasks: Tuple[TaskSpec, ...] = tuple(tasks)
        self.task_names: Tuple[str, ...] = tuple(t.name for t in self.tasks)

        # ----- Shared trunk ------------------------------------------------
        layers: List[nn.Module] = []
        prev = input_dim
        for h in shared_hidden_dims:
            layers.append(_mlp_block(prev, h, dropout))
            prev = h
        self.shared = nn.Sequential(*layers)
        shared_out_dim = prev

        # ----- Task-specific heads ----------------------------------------
        # Wrapped in ModuleDict so we can address them by name in the loss
        # function and in logs without relying on positional ordering.
        self.heads = nn.ModuleDict(
            {t.name: _TaskHead(shared_out_dim, head_hidden_dim, dropout) for t in self.tasks}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return raw outputs (logits for classification, scalars for regression).

        Returning a dict keyed by task name keeps downstream code (loss,
        metrics) self-documenting and tolerant of task-list reordering.
        """
        shared = self.shared(x)
        return {name: head(shared) for name, head in self.heads.items()}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference helper: applies sigmoid to classification heads."""
        self.eval()
        raw = self.forward(x)
        out: Dict[str, torch.Tensor] = {}
        for spec in self.tasks:
            y = raw[spec.name]
            out[spec.name] = torch.sigmoid(y) if spec.task_type == "classification" else y
        return out
