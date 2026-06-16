from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from loopdistill.data.collate import trajectory_collate
from loopdistill.data.dataset import TrajectoryDataset


class TrajectoryDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_manifest: str,
        val_manifest: str | None = None,
        test_manifest: str | None = None,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest or train_manifest
        self.test_manifest = test_manifest or self.val_manifest
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = TrajectoryDataset(Path(self.train_manifest))
            self.val_dataset = TrajectoryDataset(Path(self.val_manifest))
        if stage in (None, "test"):
            self.test_dataset = TrajectoryDataset(Path(self.test_manifest))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=trajectory_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=trajectory_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=trajectory_collate,
        )
