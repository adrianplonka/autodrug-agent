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
    """Linear -> LayerNorm -> ReLU -> Dropout."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
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

    @torch.no_grad()
    def predict_mc_dropout(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Monte Carlo Dropout: ``n_samples`` stochastic forward passes.

        Keeps the whole network in eval mode *except* the Dropout layers,
        which are re-enabled so each pass samples a different sub-network.
        The spread across passes is an estimate of the model's (epistemic)
        uncertainty.

        Works on a full batch: returns ``{task: (mean, std)}`` where ``mean``
        and ``std`` are tensors of shape ``(B,)`` aligned with the input rows.
        Classification heads are sigmoid'd before aggregation, matching
        ``predict``. ``std`` uses the population definition so a single sample
        yields 0 rather than NaN.
        """
        was_training = self.training
        self.eval()
        # Re-enable only the Dropout layers; BatchNorm/LayerNorm stay in eval.
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        # Collect per-pass outputs, then stack to (n_samples, B) per task.
        samples: Dict[str, List[torch.Tensor]] = {spec.name: [] for spec in self.tasks}
        for _ in range(max(1, n_samples)):
            raw = self.forward(x)
            for spec in self.tasks:
                val = raw[spec.name]
                if spec.task_type == "classification":
                    val = torch.sigmoid(val)
                samples[spec.name].append(val.detach().float().cpu())

        # Restore the module's original train/eval state (dropout off again).
        self.train(was_training)

        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for spec in self.tasks:
            stacked = torch.stack(samples[spec.name], dim=0)  # (n_samples, B)
            out[spec.name] = (stacked.mean(dim=0), stacked.std(dim=0, unbiased=False))
        return out