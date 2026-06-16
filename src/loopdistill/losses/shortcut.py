from __future__ import annotations

import torch
from torch import nn

from loopdistill.losses.common import masked_mse


class ShortcutConsistencyLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        student: nn.Module,
        z: torch.Tensor,
        t: torch.Tensor,
        delta: torch.Tensor,
        tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        half = 0.5 * delta
        direct = student(z, t, delta, tokens=tokens, attention_mask=attention_mask, mode="flow_map").z_next
        first = student(z, t, half, tokens=tokens, attention_mask=attention_mask, mode="flow_map").z_next
        second = student(
            first.detach(),
            t + half,
            half,
            tokens=tokens,
            attention_mask=attention_mask,
            mode="flow_map",
        ).z_next
        loss = masked_mse(direct, second.detach(), attention_mask)
        return {"loss_shortcut": self.weight * loss}
