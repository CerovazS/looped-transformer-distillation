from __future__ import annotations

from typing import Any

import torch
from torch import nn

from loopdistill.losses.common import masked_kl, masked_mse


class LoopDistillationLoss(nn.Module):
    def __init__(
        self,
        fm_weight: float = 1.0,
        endpoint_kl_weight: float = 0.2,
        latent_reconstruction_weight: float = 1.0,
        stability_weight: float = 0.05,
        temperature: float = 2.0,
        rollout_steps: int | None = None,
        mask_value: int = 0,
    ):
        super().__init__()
        self.fm_weight = fm_weight
        self.endpoint_kl_weight = endpoint_kl_weight
        self.latent_reconstruction_weight = latent_reconstruction_weight
        self.stability_weight = stability_weight
        self.temperature = temperature
        self.rollout_steps = rollout_steps
        self.mask_value = mask_value

    def _sample_pairs(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, depth, _, _ = z.shape
        device = z.device
        if depth < 2:
            raise ValueError("Trajectory z must contain at least two states.")
        a = torch.randint(0, depth - 1, (batch,), device=device)
        b = torch.randint(1, depth, (batch,), device=device)
        b = torch.maximum(b, a + 1)
        return a, b

    def _gather_depth(self, z: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, dim = z.shape
        idx = indices.reshape(batch, 1, 1, 1).expand(batch, 1, seq_len, dim)
        return z.gather(1, idx).squeeze(1)

    def _rollout(self, student: nn.Module, batch: dict[str, Any], steps: int) -> torch.Tensor:
        z = batch["z"][:, 0]
        tokens = batch["tokens"]
        mask = batch["attention_mask"]
        for step in range(steps):
            t = torch.full((z.shape[0],), step / max(steps, 1), device=z.device, dtype=z.dtype)
            delta = torch.full((z.shape[0],), 1.0 / max(steps, 1), device=z.device, dtype=z.dtype)
            out = student(z, t, delta, tokens=tokens, attention_mask=mask, mode="velocity")
            z = z + delta.reshape(-1, 1, 1) * out.velocity
        return z

    def compute(self, batch: dict[str, Any], student: nn.Module) -> dict[str, torch.Tensor]:
        z = batch["z"]
        mask = batch["attention_mask"]
        tokens = batch["tokens"]
        K = z.shape[1] - 1
        a, b = self._sample_pairs(z)
        z_a = self._gather_depth(z, a)
        z_b = self._gather_depth(z, b)
        s = torch.rand(z.shape[0], device=z.device, dtype=z.dtype)
        t_a = a.to(z.dtype) / max(K, 1)
        t_b = b.to(z.dtype) / max(K, 1)
        t = t_a + s * (t_b - t_a)
        delta = (t_b - t_a).clamp_min(1e-6)
        z_t = (1 - s.reshape(-1, 1, 1)) * z_a + s.reshape(-1, 1, 1) * z_b
        target_v = (z_b - z_a) / delta.reshape(-1, 1, 1)

        out = student(z_t, t, delta, tokens=tokens, attention_mask=mask, mode="velocity")
        fm_loss = masked_mse(out.velocity, target_v, mask)

        rollout_steps = self.rollout_steps or K
        z_hat_K = self._rollout(student, batch, rollout_steps)
        latent_reconstruction = masked_mse(z_hat_K, z[:, -1], mask)

        endpoint_kl = z.new_zeros(())
        logits = batch.get("logits")
        if logits is not None and out.logits is not None:
            endpoint_out = student(
                z_hat_K,
                torch.ones(z.shape[0], device=z.device, dtype=z.dtype),
                torch.zeros(z.shape[0], device=z.device, dtype=z.dtype),
                tokens=tokens,
                attention_mask=mask,
            )
            endpoint_kl = masked_kl(endpoint_out.logits, logits[:, -1], mask, self.temperature)

        stability = z.new_zeros(())
        flow_out = student(
            z[:, -1],
            torch.ones(z.shape[0], device=z.device, dtype=z.dtype),
            torch.zeros(z.shape[0], device=z.device, dtype=z.dtype),
            tokens=tokens,
            attention_mask=mask,
            mode="flow_map",
        )
        if flow_out.z_next is not None:
            stability = masked_mse(flow_out.z_next, z[:, -1], mask)

        total = (
            self.fm_weight * fm_loss
            + self.endpoint_kl_weight * endpoint_kl
            + self.latent_reconstruction_weight * latent_reconstruction
            + self.stability_weight * stability
        )
        latent_mse_per_depth = ((z[:, 1:] - z[:, :-1]).pow(2).mean()).detach()
        return {
            "loss": total,
            "loss_fm": fm_loss.detach(),
            "loss_endpoint_kl": endpoint_kl.detach(),
            "loss_latent_reconstruction": latent_reconstruction.detach(),
            "loss_stability": stability.detach(),
            "metric_latent_step_mse": latent_mse_per_depth,
            "metric_fixed_point_residual": batch["residual_norm"].float().mean().detach(),
            "metric_solver_iters": batch["solver_iters"].float().mean().detach(),
        }

    forward = compute
