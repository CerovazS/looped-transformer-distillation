from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from loopdistill.data.schema import TrajectoryRecord


class TrajectoryDataset(Dataset[dict[str, Any]]):
    """Dataset backed by a JSONL manifest and row-indexed Torch shards."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Trajectory manifest not found: {self.manifest_path}. "
                "Run loopdistill-extract first or point data.*_manifest to an existing file."
            )
        self.records = [
            TrajectoryRecord.from_json(json.loads(line))
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._shard_cache: dict[Path, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_shard(self, path: Path) -> dict[str, Any]:
        if path not in self._shard_cache:
            self._shard_cache[path] = torch.load(path, map_location="cpu")
        return self._shard_cache[path]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        shard = self._load_shard(record.resolve_shard(self.manifest_path))
        row = record.row
        logits = shard.get("logits")
        item = {
            "tokens": shard["tokens"][row],
            "attention_mask": shard["attention_mask"][row],
            "teacher_id": record.teacher_id,
            "K": torch.as_tensor(record.K, dtype=torch.long),
            "z": shard["z"][row],
            "logits": None if logits is None else logits[row],
            "loss_K": shard["loss_K"][row],
            "residual_norm": shard["residual_norm"][row],
            "solver_iters": shard["solver_iters"][row],
            "sample_id": record.sample_id,
            "metadata": record.metadata,
        }
        return item
