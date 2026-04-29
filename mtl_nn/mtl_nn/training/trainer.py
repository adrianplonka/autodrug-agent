"""Training loop for the MTL ADMET network."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import EPOCHS, LEARNING_RATE, WEIGHT_DECAY, TaskSpec
from ..model import ADMET_MTL_Network
from .loss import MaskedMultiTaskLoss


class _RunningMean:
    """Numerically-cheap running average for log-friendly per-task losses."""

    def __init__(self) -> None:
        self.sum: float = 0.0
        self.n: int = 0

    def update(self, value: float, count: int = 1) -> None:
        self.sum += value * count
        self.n += count

    @property
    def value(self) -> float:
        return self.sum / self.n if self.n else 0.0


class Trainer:
    def __init__(
        self,
        model: ADMET_MTL_Network,
        tasks: Sequence[TaskSpec],
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tasks = tuple(tasks)
        self.criterion = MaskedMultiTaskLoss(self.tasks).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------ epoch
    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_meter = _RunningMean()
        per_task_meters: Dict[str, _RunningMean] = {t.name: _RunningMean() for t in self.tasks}

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for fps, targets, mask in loader:
                fps = fps.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                preds = self.model(fps)
                total_loss, per_task = self.criterion(preds, targets, mask)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    self.optimizer.step()

                bs = fps.size(0)
                total_meter.update(total_loss.item(), bs)
                for name, value in per_task.items():
                    per_task_meters[name].update(value.item(), bs)

        log: Dict[str, float] = {"total": total_meter.value}
        log.update({name: m.value for name, m in per_task_meters.items()})
        return log

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = EPOCHS,
    ) -> None:
        for epoch in tqdm(range(1, epochs + 1), desc="epochs"):
            train_log = self._run_epoch(train_loader, train=True)
            val_log = self._run_epoch(val_loader, train=False)
            self._print_epoch(epoch, train_log, val_log)

    def _print_epoch(self, epoch: int, train_log: Dict[str, float], val_log: Dict[str, float]) -> None:
        head = f"[epoch {epoch:>3d}] train_total={train_log['total']:.4f} val_total={val_log['total']:.4f}"
        per_task_str = "  ".join(
            f"{t.name}: tr={train_log[t.name]:.4f}/va={val_log[t.name]:.4f}"
            for t in self.tasks
        )
        print(f"{head}  ::  {per_task_str}")
