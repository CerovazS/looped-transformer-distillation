from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch import nn

from loopdistill.teachers.base import TeacherRunner


class DistillationModule(L.LightningModule):
    def __init__(
        self,
        student: nn.Module,
        loss_module: nn.Module,
        quality_evaluator: nn.Module | None = None,
        teacher: TeacherRunner | None = None,
        live_depths: list[int] | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        metrics_dir: str | None = None,
        test_metric_prefix: str = "test",
    ):
        super().__init__()
        self.student = student
        self.loss_module = loss_module
        self.quality_evaluator = quality_evaluator
        self.teacher = teacher
        self.live_depths = live_depths
        self.lr = lr
        self.weight_decay = weight_decay
        self.metrics_dir = Path(metrics_dir) if metrics_dir else None
        self.test_metric_prefix = str(test_metric_prefix)
        self._teacher_student_connected = False

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _step(self, batch: dict[str, Any], prefix: str) -> torch.Tensor:
        batch = self._prepare_batch(batch, prefix)
        return self._loss_step(batch, prefix)

    def _prepare_batch(self, batch: dict[str, Any], prefix: str) -> dict[str, Any]:
        batch = self._move_batch(batch)
        return self._maybe_add_live_teacher_trajectory(batch, return_logits=self._needs_teacher_logits(prefix))

    def _loss_step(self, batch: dict[str, Any], prefix: str) -> torch.Tensor:
        metrics = self.loss_module.compute(batch, self.student)
        for key, value in metrics.items():
            self.log(
                f"{prefix}/{key}",
                value,
                prog_bar=key == "loss",
                on_step=prefix == "train",
                on_epoch=True,
                sync_dist=self._sync_dist(),
            )
        return metrics["loss"]

    def _quality_step(self, batch: dict[str, Any], prefix: str, *, stage: str | None = None) -> None:
        if self.quality_evaluator is None or not bool(getattr(self.quality_evaluator, "enabled", False)):
            return
        stage = stage or prefix
        if stage == "val" and not self._should_run_val_quality():
            return
        if stage == "test" and not bool(getattr(self.quality_evaluator, "run_on_test", True)):
            return
        metrics = self.quality_evaluator.compute(batch, self.student, teacher=self.teacher)
        for key, value in metrics.items():
            self.log(
                f"eval_quality/{prefix}/{key}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=self._sync_dist(),
            )

    def _should_run_val_quality(self) -> bool:
        every_n_epochs = int(getattr(self.quality_evaluator, "every_n_epochs", 1))
        if every_n_epochs <= 0:
            return False
        return (int(self.current_epoch) + 1) % every_n_epochs == 0

    def _sync_dist(self) -> bool:
        try:
            return int(self.trainer.world_size) > 1
        except RuntimeError:
            return False

    def _needs_teacher_logits(self, prefix: str) -> bool:
        if prefix == "train":
            return float(getattr(self.loss_module, "endpoint_kl_weight", 0.0)) != 0.0
        quality_enabled = self.quality_evaluator is not None and bool(getattr(self.quality_evaluator, "enabled", False))
        if prefix == "val":
            quality_enabled = quality_enabled and self._should_run_val_quality()
        if prefix == "test":
            quality_enabled = quality_enabled and bool(getattr(self.quality_evaluator, "run_on_test", True))
        quality_needs_batch_logits = bool(
            quality_enabled
            and getattr(self.quality_evaluator, "needs_batch_teacher_logits", lambda teacher=None: True)(self.teacher)
        )
        return quality_needs_batch_logits or float(getattr(self.loss_module, "endpoint_kl_weight", 0.0)) != 0.0

    def _maybe_add_live_teacher_trajectory(
        self,
        batch: dict[str, Any],
        *,
        return_logits: bool | None = None,
    ) -> dict[str, Any]:
        if "z" in batch:
            return batch
        if self.teacher is None:
            raise KeyError(
                "Batch does not contain trajectory key 'z'. "
                "Use an offline TrajectoryDataModule or enable live.teacher."
            )
        depths = self.live_depths
        if depths is None:
            max_depth = int(getattr(self.teacher, "max_depth", 4))
            depths = list(range(max_depth + 1))
        previous_return_logits = getattr(self.teacher, "return_logits", None)
        if return_logits is not None and previous_return_logits is not None:
            self.teacher.return_logits = bool(return_logits)
        try:
            with torch.no_grad():
                output = self.teacher.run_batch(
                    tokens=batch["tokens"],
                    attention_mask=batch["attention_mask"],
                    depths=[int(depth) for depth in depths],
                )
        finally:
            if previous_return_logits is not None:
                self.teacher.return_logits = previous_return_logits
        batch["K"] = torch.full(
            (batch["tokens"].shape[0],),
            output.z.shape[1] - 1,
            dtype=torch.long,
            device=output.z.device,
        )
        batch["z"] = output.z.detach().clone()
        batch["logits"] = None if output.logits is None else output.logits.detach().clone()
        batch["loss_K"] = output.loss_K.detach().clone()
        batch["residual_norm"] = output.residual_norm.detach().clone()
        batch["solver_iters"] = output.solver_iters.detach().clone()
        batch["teacher_id"] = [self.teacher.teacher_id] * int(batch["tokens"].shape[0])
        return batch

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._prepare_batch(batch, "val")
        loss = self._loss_step(batch, "val")
        self._quality_step(batch, "val")
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._prepare_batch(batch, "test")
        loss = self._loss_step(batch, self.test_metric_prefix)
        self._quality_step(batch, self.test_metric_prefix, stage="test")
        return loss

    def configure_optimizers(self):
        self._connect_live_teacher_student()
        return torch.optim.AdamW(self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def configure_model(self) -> None:
        self._connect_live_teacher_student()

    def on_fit_start(self) -> None:
        self._connect_live_teacher_student()

    def on_test_start(self) -> None:
        self._connect_live_teacher_student()

    def _connect_live_teacher_student(self) -> None:
        if self._teacher_student_connected:
            return
        self._place_live_teacher()
        if self.teacher is not None and hasattr(self.student, "initialize_from_teacher"):
            self.student.initialize_from_teacher(self.teacher)
        self._teacher_student_connected = True

    def _place_live_teacher(self) -> None:
        if self.teacher is not None and hasattr(self.teacher, "set_device"):
            self.teacher.set_device(self.device)

    def on_train_epoch_end(self) -> None:
        self._append_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._append_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._append_metrics(self.test_metric_prefix)

    def _append_metrics(self, split: str) -> None:
        if self.metrics_dir is None or self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        path = self.metrics_dir / f"{split}.csv"
        row = {"epoch": self.current_epoch}
        for key, value in self.trainer.callback_metrics.items():
            if key.startswith(f"{split}/") or key.startswith(f"eval_quality/{split}/"):
                row[key] = float(value.detach().cpu())
        if torch.cuda.is_available() and self.device.type == "cuda":
            row[f"{split}/metric_cuda_memory_allocated_peak_mb"] = (
                torch.cuda.max_memory_allocated(self.device) / (1024**2)
            )
            row[f"{split}/metric_cuda_memory_reserved_peak_mb"] = (
                torch.cuda.max_memory_reserved(self.device) / (1024**2)
            )
        if len(row) == 1:
            return
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
