from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TrajectoryRecord:
    sample_id: str
    teacher_id: str
    teacher_repo: str | None
    teacher_ckpt: str | None
    tokenizer_id: str | None
    dataset_id: str
    split: str
    shard_path: str
    row: int
    seq_len: int
    K: int
    dtype: str
    latent_shape: list[int]
    logit_shape: list[int] | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TrajectoryRecord":
        return cls(**payload)

    def resolve_shard(self, manifest_path: str | Path) -> Path:
        shard = Path(self.shard_path)
        if shard.is_absolute():
            return shard
        return Path(manifest_path).parent / shard


@dataclass
class TrajectoryBatch:
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    teacher_id: list[str]
    K: torch.Tensor
    z: torch.Tensor
    logits: torch.Tensor | None
    loss_K: torch.Tensor
    residual_norm: torch.Tensor
    solver_iters: torch.Tensor
    sample_id: list[str] | None = None
    metadata: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "attention_mask": self.attention_mask,
            "teacher_id": self.teacher_id,
            "K": self.K,
            "z": self.z,
            "logits": self.logits,
            "loss_K": self.loss_K,
            "residual_norm": self.residual_norm,
            "solver_iters": self.solver_iters,
            "sample_id": self.sample_id,
            "metadata": self.metadata,
        }
