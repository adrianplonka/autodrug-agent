"""Masked multi-task loss.

The CRITICAL requirement: each molecule has labels for *some* of the 5 tasks
but is missing labels for the rest. We must not let those missing entries
contribute gradients — otherwise the network would learn from arbitrary
placeholder values (NaN-replaced-with-0 in the dataset).

Masking strategy
----------------
For each task k:
  * ``per_sample_loss_k`` is computed with reduction='none', giving one loss
    value per sample in the batch.
  * We multiply by ``mask[:, k]`` (1 if the label exists, 0 otherwise),
    zeroing out contributions from samples that lack a ground-truth label
    for task k. This is mathematically equivalent to dropping those samples
    *only for task k*, while keeping them in the batch for the other tasks
    where they DO have labels.
  * We average over the *number of valid (unmasked) samples for task k*,
    not the full batch size — otherwise tasks with sparse labels would have
    artificially shrunken losses and would be ignored by the optimizer.
  * If a batch happens to contain zero labeled samples for some task, that
    task contributes 0 to the total loss for that step (no gradient flows
    through its head, which is the correct behavior — there is nothing to
    learn from).

The total loss is the *sum* of the per-task losses. (Equal weighting; this
is the standard hard-parameter-sharing baseline. Per-task weights / GradNorm
/ uncertainty weighting could plug in here if desired.)
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TaskSpec


class MaskedMultiTaskLoss(nn.Module):
    def __init__(self, tasks: Sequence[TaskSpec]) -> None:
        super().__init__()
        self.tasks: Tuple[TaskSpec, ...] = tuple(tasks)
        # Index lookup so we can pull the right column of the mask/target tensors
        # for each named task head.
        self._task_index: Dict[str, int] = {t.name: i for i, t in enumerate(self.tasks)}

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

            # Per-sample loss WITHOUT reduction so we can apply the mask
            # element-wise. BCEWithLogitsLoss is numerically stabler than
            # sigmoid + BCE; for regression we use MSE.
            if spec.task_type == "classification":
                per_sample = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
            else:
                per_sample = F.mse_loss(y_pred, y_true, reduction="none")

            # Zero out contributions from samples lacking a label for this task.
            masked = per_sample * m

            # Normalize by the number of *valid* (labeled) samples, not the
            # batch size. If a batch has zero labels for this task, fall back
            # to 0 — clamp avoids division-by-zero without injecting bias
            # (numerator is 0 anyway in that case).
            n_valid = m.sum().clamp(min=1.0)
            task_loss = masked.sum() / n_valid

            total = total + task_loss
            per_task_log[spec.name] = task_loss.detach()

        return total, per_task_log
