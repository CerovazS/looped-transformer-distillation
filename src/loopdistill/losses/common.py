from __future__ import annotations

import torch
import torch.nn.functional as F


def expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while mask.dim() < target.dim():
        mask = mask.unsqueeze(-1)
    return mask.to(dtype=target.dtype, device=target.device)


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    m = expand_mask(mask, x)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return masked_mean((pred - target).pow(2), mask)


def masked_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    t = temperature
    log_p = F.log_softmax(student_logits / t, dim=-1)
    q = F.softmax(teacher_logits / t, dim=-1)
    kl = F.kl_div(log_p, q, reduction="none").sum(dim=-1) * (t * t)
    return masked_mean(kl, mask)
