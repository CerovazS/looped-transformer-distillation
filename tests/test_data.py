from __future__ import annotations

import torch

from loopdistill.data.collate import trajectory_collate
from loopdistill.data.dataset import TrajectoryDataset
from loopdistill.data.writer import TrajectoryShardWriter


def test_shard_roundtrip(tmp_path):
    writer = TrajectoryShardWriter(tmp_path, teacher_id="mock", dataset_id="unit")
    tensors = {
        "tokens": torch.ones(2, 5, dtype=torch.long),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "K": torch.full((2,), 3),
        "z": torch.randn(2, 4, 5, 8),
        "logits": torch.randn(2, 4, 5, 16),
        "loss_K": torch.zeros(2),
        "residual_norm": torch.zeros(2),
        "solver_iters": torch.full((2,), 3.0),
    }
    writer.write_shard(tensors, shard_name="shard.pt")
    dataset = TrajectoryDataset(writer.manifest_path)
    assert len(dataset) == 2
    batch = trajectory_collate([dataset[0], dataset[1]])
    assert batch["z"].shape == (2, 4, 5, 8)
    assert batch["logits"].shape == (2, 4, 5, 16)
