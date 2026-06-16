from __future__ import annotations

from pathlib import Path

import torch

from loopdistill.data.token_shards import TokenShardDataModule, TokenShardDataset


def test_token_shard_dataset_roundtrip(tmp_path: Path):
    path = tmp_path / "train.pt"
    tokens = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    torch.save({"tokens": tokens}, path)

    dataset = TokenShardDataset(str(path))

    assert len(dataset) == 4
    item = dataset[2]
    assert item["tokens"].shape == (6,)
    assert item["attention_mask"].all()


def test_token_shard_datamodule_batches(tmp_path: Path):
    for split in ("train", "val", "test"):
        torch.save({"tokens": torch.arange(24, dtype=torch.int32).reshape(4, 6)}, tmp_path / f"{split}.pt")
    data = TokenShardDataModule(
        train_path=str(tmp_path / "train.pt"),
        val_path=str(tmp_path / "val.pt"),
        test_path=str(tmp_path / "test.pt"),
        batch_size=2,
    )

    data.setup("fit")
    batch = next(iter(data.train_dataloader()))

    assert batch["tokens"].shape == (2, 6)
    assert batch["attention_mask"].shape == (2, 6)

