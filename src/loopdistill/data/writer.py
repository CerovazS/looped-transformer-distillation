from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from loopdistill.data.schema import TrajectoryRecord


class TrajectoryShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        teacher_id: str,
        teacher_repo: str | None = None,
        teacher_ckpt: str | None = None,
        tokenizer_id: str | None = None,
        dataset_id: str = "unknown",
        split: str = "train",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.teacher_id = teacher_id
        self.teacher_repo = teacher_repo
        self.teacher_ckpt = teacher_ckpt
        self.tokenizer_id = tokenizer_id
        self.dataset_id = dataset_id
        self.split = split
        self.manifest_path = self.output_dir / "manifest.jsonl"

    def write_shard(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        shard_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        required = {"tokens", "attention_mask", "K", "z", "loss_K", "residual_norm", "solver_iters"}
        missing = required.difference(tensors)
        if missing:
            raise KeyError(f"Trajectory shard missing required keys: {sorted(missing)}")

        shard_path = self.output_dir / shard_name
        if shard_path.exists():
            raise FileExistsError(f"Refusing to overwrite trajectory shard: {shard_path}")
        torch.save(tensors, shard_path)

        metadata = metadata or {}
        n_rows = int(tensors["tokens"].shape[0])
        K = int(tensors["z"].shape[1] - 1)
        latent_shape = list(tensors["z"].shape[1:])
        logits = tensors.get("logits")
        logit_shape = None if logits is None else list(logits.shape[1:])
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            for row in range(n_rows):
                seq_len = int(tensors["attention_mask"][row].sum().item())
                record = TrajectoryRecord(
                    sample_id=f"{self.split}-{shard_path.stem}-{row}",
                    teacher_id=self.teacher_id,
                    teacher_repo=self.teacher_repo,
                    teacher_ckpt=self.teacher_ckpt,
                    tokenizer_id=self.tokenizer_id,
                    dataset_id=self.dataset_id,
                    split=self.split,
                    shard_path=shard_path.name,
                    row=row,
                    seq_len=seq_len,
                    K=K,
                    dtype=str(tensors["z"].dtype).replace("torch.", ""),
                    latent_shape=latent_shape,
                    logit_shape=logit_shape,
                    metadata=metadata,
                )
                handle.write(json.dumps(record.__dict__) + "\n")
        return shard_path
