from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch
from torch import nn

from loopdistill.losses.common import masked_mean


class MeanFlowLoss(nn.Module):
    def __init__(
        self,
        data_proportion: float = 0.75,
        p_mean: float = -0.4,
        p_std: float = 1.0,
        norm_eps: float = 0.01,
        norm_p: float = 1.0,
        jvp_autocast_enabled: bool = False,
        fallback: str = "finite_difference",
        finite_difference_eps: float = 1e-3,
    ):
        super().__init__()
        self.data_proportion = data_proportion
        self.p_mean = p_mean
        self.p_std = p_std
        self.norm_eps = norm_eps
        self.norm_p = norm_p
        self.jvp_autocast_enabled = jvp_autocast_enabled
        self.fallback = fallback
        self.finite_difference_eps = finite_difference_eps

    def sample_tr(self, batch: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.sigmoid(torch.randn(batch, device=device, dtype=dtype) * self.p_std + self.p_mean)
        r = torch.sigmoid(torch.randn(batch, device=device, dtype=dtype) * self.p_std + self.p_mean)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        local_count = int(batch * self.data_proportion)
        if local_count:
            r[:local_count] = t[:local_count]
        return t, r

    def _finite_difference_jvp(
        self,
        fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        z: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = self.finite_difference_eps
        base = fn(z, t, r)
        shifted = fn(z + eps * v, t + eps, r)
        return base, (shifted - base) / eps

    def forward(
        self,
        clean: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = clean.shape[0]
        t, r = self.sample_tr(batch, clean.device, clean.dtype)
        e = torch.randn_like(clean)
        z_t = (1 - t.reshape(batch, 1, 1)) * clean + t.reshape(batch, 1, 1) * e
        v = e - clean

        def u_func(z: torch.Tensor, t_in: torch.Tensor, r_in: torch.Tensor) -> torch.Tensor:
            return model(z, t_in, t_in - r_in)

        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=False)
            if clean.is_cuda and not self.jvp_autocast_enabled
            else nullcontext()
        )
        with autocast_ctx:
            try:
                u, du_dt = torch.func.jvp(
                    u_func,
                    (z_t, t, r),
                    (v, torch.ones_like(t), torch.zeros_like(r)),
                )
            except RuntimeError:
                if self.fallback != "finite_difference":
                    raise
                u, du_dt = self._finite_difference_jvp(u_func, z_t, t, r, v)
            target = (v - (t - r).reshape(batch, 1, 1) * du_dt).detach()
            per_token = (u - target).pow(2).mean(dim=-1)
            per_sample = masked_mean(per_token, mask).reshape(())
            adaptive = (per_sample.detach() + self.norm_eps) ** self.norm_p
            loss = per_sample / adaptive
        return {"loss_meanflow": loss, "meanflow_interval": (t - r).mean().detach()}
