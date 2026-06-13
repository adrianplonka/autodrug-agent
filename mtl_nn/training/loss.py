"""Masked multi-task loss.

The CRITICAL requirement: each molecule has labels for *some* of the tasks
but is missing labels for the rest. We must not let those missing entries
contribute gradients — otherwise the network would learn from arbitrary
placeholder values (NaN-replaced-with-0 in the dataset).

Masking strategy
----------------
For each task k:
  * ``per_sample_loss_k`` is computed with reduction='none', giving one loss
    value per sample in the batch.
  * We multiply by ``mask[:, k]`` (1 if the label exists, 0 otherwise).
  * We average over the *number of valid (unmasked) samples for task k*,
    not the full batch size — otherwise tasks with sparse labels would have
    artificially shrunken losses and would be ignored by the optimizer.
  * If a batch happens to contain zero labeled samples for some task, that
    task contributes 0 to the total loss for that step.

The total loss is the weighted sum of per-task losses (weights from TASK_WEIGHTS).
Classification heads use BCEWithLogitsLoss with per-task pos_weight for class
imbalance. Regression heads use Huber loss (delta=1.0).
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TASK_WEIGHTS, TaskSpec


class MaskedMultiTaskLoss(nn.Module):
    def __init__(self, tasks: Sequence[TaskSpec]) -> None:
        super().__init__()
        self.tasks: Tuple[TaskSpec, ...] = tuple(tasks)
        self._task_index: Dict[str, int] = {t.name: i for i, t in enumerate(self.tasks)}

        # n_negative / n_positive for imbalanced classification tasks
        self._pos_weights: Dict[str, float] = {
            "bbb":    0.31,   # 465/1500
            "cyp3a4": 1.41,   # 7218/5110
            "herg":   0.45,   # 201/444
        }

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],   # name -> (B,) raw outputs
        targets: torch.Tensor,                  # (B, n_tasks) — NaNs already replaced with 0
        mask: torch.Tensor,                     # (B, n_tasks) — 1 where labeled
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss + a dict of per-task losses (detached, for logging)."""
        total = torch.zeros((), device=targets.device)
        per_task_log: Dict[str, torch.Tensor] = {}

        for spec in self.tasks:
            k = self._task_index[spec.name]
            y_pred = predictions[spec.name]      # (B,) raw logits or scalars
            y_true = targets[:, k]               # (B,)
            m = mask[:, k]                       # (B,) 1 if labeled, else 0

            if spec.task_type == "classification":
                pw = self._pos_weights.get(spec.name, 1.0)
                pw_tensor = torch.tensor(pw, device=y_pred.device)
                per_sample = F.binary_cross_entropy_with_logits(
                    y_pred, y_true, pos_weight=pw_tensor, reduction="none"
                )
            else:
                per_sample = F.huber_loss(y_pred, y_true, reduction="none", delta=1.0)

            masked = per_sample * m
            n_valid = m.sum().clamp(min=1.0)
            task_loss = masked.sum() / n_valid

            weight = TASK_WEIGHTS.get(spec.name, 1.0)
            total = total + weight * task_loss
            per_task_log[spec.name] = task_loss.detach()

        return total, per_task_log
