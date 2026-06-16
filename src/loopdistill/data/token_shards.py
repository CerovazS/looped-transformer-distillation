from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


class TokenShardDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(f"Token shard not found: {self.path}")
        data = torch.load(self.path, map_location="cpu", weights_only=True)
        self.tokens = data["tokens"].long()
        self.attention_mask = data.get("attention_mask")
        if self.attention_mask is None:
            self.attention_mask = torch.ones_like(self.tokens, dtype=torch.bool)
        else:
            self.attention_mask = self.attention_mask.bool()
        if self.tokens.ndim != 2:
            raise ValueError(f"Expected tokens with shape [N,L], got {tuple(self.tokens.shape)}.")
        if self.attention_mask.shape != self.tokens.shape:
            raise ValueError("attention_mask must have the same shape as tokens.")

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": self.tokens[index],
            "attention_mask": self.attention_mask[index],
        }


class TokenShardDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        train_path: str,
        val_path: str,
        test_path: str | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path or val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = TokenShardDataset(self.train_path)
            self.val_dataset = TokenShardDataset(self.val_path)
        if stage in (None, "test"):
            self.test_dataset = TokenShardDataset(self.test_path)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False, drop_last=False)

    def _loader(self, dataset: TokenShardDataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
        )

