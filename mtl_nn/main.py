"""Entry point: download data, build the MTL model, train, and report.

Run from the parent directory:

    python -m mtl_nn.main
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score

from .config import EPOCHS, RANDOM_SEED, TASKS, configure_cuda_runtime, resolve_device
from .data import build_dataloaders, download_and_merge_tdc
from .model import ADMET_MTL_Network
from .training import Trainer


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: ADMET_MTL_Network,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """Compute per-task metrics on a loader, respecting the label mask."""
    model.eval()
    # Collect predictions and targets per task, dropping rows that lack a label.
    bucket: dict[str, list[tuple[float, float]]] = {t.name: [] for t in model.tasks}
    for fps, targets, mask in loader:
        fps = fps.to(device)
        preds = model.predict(fps)  # already sigmoid'd for classification heads
        for k, spec in enumerate(model.tasks):
            m = mask[:, k].bool()
            if not m.any():
                continue
            y_pred = preds[spec.name].detach().cpu()[m].numpy()
            y_true = targets[:, k][m].numpy()
            bucket[spec.name].extend(zip(y_pred.tolist(), y_true.tolist()))

    print("\n[test] per-task metrics:")
    for spec in model.tasks:
        if not bucket[spec.name]:
            print(f"  - {spec.name}: no labels in this split")
            continue
        y_pred, y_true = map(np.array, zip(*bucket[spec.name]))
        if spec.task_type == "classification":
            try:
                metric = roc_auc_score(y_true, y_pred)
                print(f"  - {spec.name:<10s} ROC-AUC = {metric:.4f}  (n={len(y_true)})")
            except ValueError:
                # Single-class split — AUC is undefined.
                print(f"  - {spec.name:<10s} ROC-AUC = n/a (single class)  (n={len(y_true)})")
        else:
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            print(f"  - {spec.name:<10s} RMSE    = {rmse:.4f}  (n={len(y_true)})")


def main() -> None:
    _set_seed(RANDOM_SEED)
    configure_cuda_runtime()
    device = resolve_device()
    print(f"[main] resolved device: {device}")

    df = download_and_merge_tdc(TASKS)
    train_loader, val_loader, test_loader = build_dataloaders(df, TASKS)

    model = ADMET_MTL_Network(TASKS)
    print(model)

    trainer = Trainer(model, TASKS, device=device)
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    evaluate(model, test_loader, trainer.device)


if __name__ == "__main__":
    main()
