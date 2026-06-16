from __future__ import annotations

from pathlib import Path

import lightning as L
import torch

from loopdistill.data.datamodule import TrajectoryDataModule
from loopdistill.data.writer import TrajectoryShardWriter
from loopdistill.losses.p0 import LoopDistillationLoss
from loopdistill.models.student import StudentFlowModel
from loopdistill.train_module import DistillationModule


def test_lightning_two_batches(tmp_path: Path):
    writer = TrajectoryShardWriter(tmp_path, teacher_id="mock", dataset_id="unit")
    tensors = {
        "tokens": torch.randint(0, 16, (8, 6)),
        "attention_mask": torch.ones(8, 6, dtype=torch.bool),
        "K": torch.full((8,), 3),
        "z": torch.randn(8, 4, 6, 8),
        "logits": torch.randn(8, 4, 6, 16),
        "loss_K": torch.zeros(8),
        "residual_norm": torch.zeros(8),
        "solver_iters": torch.full((8,), 3.0),
    }
    writer.write_shard(tensors, shard_name="shard.pt")
    data = TrajectoryDataModule(str(writer.manifest_path), batch_size=4)
    student = StudentFlowModel(latent_dim=8, hidden_dim=32, num_layers=1, num_heads=4, vocab_size=16)
    module = DistillationModule(student, LoopDistillationLoss(), metrics_dir=str(tmp_path / "metrics"))
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=2,
        limit_val_batches=1,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=data)
    assert (tmp_path / "metrics" / "train.csv").exists()
