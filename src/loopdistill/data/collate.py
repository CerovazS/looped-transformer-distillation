from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _pad_1d(x: torch.Tensor, length: int, value: int = 0) -> torch.Tensor:
    return F.pad(x, (0, length - x.shape[0]), value=value)


def _pad_z(x: torch.Tensor, length: int) -> torch.Tensor:
    return F.pad(x, (0, 0, 0, length - x.shape[-2], 0, 0), value=0.0)


def _pad_logits(x: torch.Tensor, length: int) -> torch.Tensor:
    return F.pad(x, (0, 0, 0, length - x.shape[-2], 0, 0), value=0.0)


def trajectory_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(int(item["tokens"].shape[0]) for item in batch)
    tokens = torch.stack([_pad_1d(item["tokens"], max_len, 0) for item in batch])
    attention_mask = torch.stack([_pad_1d(item["attention_mask"], max_len, 0) for item in batch])
    z = torch.stack([_pad_z(item["z"], max_len) for item in batch])

    logits_items = [item["logits"] for item in batch]
    logits = None
    if all(item is not None for item in logits_items):
        logits = torch.stack([_pad_logits(item, max_len) for item in logits_items])

    return {
        "tokens": tokens.long(),
        "attention_mask": attention_mask.bool(),
        "teacher_id": [item["teacher_id"] for item in batch],
        "K": torch.stack([item["K"] for item in batch]),
        "z": z.float(),
        "logits": None if logits is None else logits.float(),
        "loss_K": torch.stack([torch.as_tensor(item["loss_K"]).float() for item in batch]),
        "residual_norm": torch.stack(
            [torch.as_tensor(item["residual_norm"]).float() for item in batch]
        ),
        "solver_iters": torch.stack([torch.as_tensor(item["solver_iters"]).float() for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
