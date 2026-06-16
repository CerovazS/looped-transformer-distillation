from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch import nn


class DistillationModule(L.LightningModule):
    def __init__(
        self,
        student: nn.Module,
        loss_module: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        metrics_dir: str | None = None,
    ):
        super().__init__()
        self.student = student
        self.loss_module = loss_module
        self.lr = lr
        self.weight_decay = weight_decay
        self.metrics_dir = Path(metrics_dir) if metrics_dir else None

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _step(self, batch: dict[str, Any], prefix: str) -> torch.Tensor:
        batch = self._move_batch(batch)
        metrics = self.loss_module.compute(batch, self.student)
        for key, value in metrics.items():
            self.log(f"{prefix}/{key}", value, prog_bar=key == "loss", on_step=prefix == "train", on_epoch=True)
        return metrics["loss"]

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def on_train_epoch_end(self) -> None:
        self._append_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._append_metrics("val")

    def _append_metrics(self, split: str) -> None:
        if self.metrics_dir is None or self.trainer.sanity_checking:
            return
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        path = self.metrics_dir / f"{split}.csv"
        row = {"epoch": self.current_epoch}
        for key, value in self.trainer.callback_metrics.items():
            if key.startswith(f"{split}/"):
                row[key] = float(value.detach().cpu())
        if len(row) == 1:
            return
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
