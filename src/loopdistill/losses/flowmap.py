from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from loopdistill.losses.common import masked_mse


def _require_trajectory(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = batch["z"]
    mask = batch["attention_mask"]
    tokens = batch["tokens"]
    if z.dim() != 4:
        raise ValueError(f"Expected batch['z'] with shape [B,K,L,D], got {tuple(z.shape)}.")
    if z.shape[1] < 2:
        raise ValueError("Trajectory z must contain at least two states.")
    return z, mask, tokens


def _gather_depth(z: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, _, seq_len, dim = z.shape
    idx = indices.reshape(batch, 1, 1, 1).expand(batch, 1, seq_len, dim)
    return z.gather(1, idx).squeeze(1)


def _normalized_time(indices: torch.Tensor, depth: int, dtype: torch.dtype) -> torch.Tensor:
    return indices.to(dtype=dtype) / max(depth - 1, 1)


def _sample_pairs(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, depth, _, _ = z.shape
    device = z.device
    a = torch.randint(0, depth - 1, (batch,), device=device)
    span = torch.empty(batch, device=device, dtype=torch.long)
    for idx in range(batch):
        span[idx] = torch.randint(1, depth - int(a[idx]), (), device=device)
    b = a + span
    return a, b


def _sample_triplets(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, depth, _, _ = z.shape
    if depth < 3:
        raise ValueError("Compositional flow-map loss requires at least three trajectory states.")
    device = z.device
    a = torch.empty(batch, device=device, dtype=torch.long)
    m = torch.empty(batch, device=device, dtype=torch.long)
    b = torch.empty(batch, device=device, dtype=torch.long)
    for idx in range(batch):
        a_i = torch.randint(0, depth - 2, (), device=device)
        b_i = torch.randint(int(a_i) + 2, depth, (), device=device)
        m_i = torch.randint(int(a_i) + 1, int(b_i), (), device=device)
        a[idx], m[idx], b[idx] = a_i, m_i, b_i
    return a, m, b


def _metric_defaults(batch: dict[str, Any], z: torch.Tensor) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {
        "metric_latent_step_mse": (z[:, 1:] - z[:, :-1]).pow(2).mean().detach(),
    }
    residual_norm = batch.get("residual_norm")
    if residual_norm is not None:
        metrics["metric_fixed_point_residual"] = residual_norm.float().mean().detach()
    solver_iters = batch.get("solver_iters")
    if solver_iters is not None:
        metrics["metric_solver_iters"] = solver_iters.float().mean().detach()
    return metrics


class LagrangianFlowMapLoss(nn.Module):
    """Discrete supervised flow-map matching on teacher trajectory pairs."""

    def __init__(
        self,
        map_weight: float = 1.0,
        velocity_weight: float = 0.0,
        stability_weight: float = 0.0,
    ):
        super().__init__()
        self.map_weight = float(map_weight)
        self.velocity_weight = float(velocity_weight)
        self.stability_weight = float(stability_weight)

    def compute(self, batch: dict[str, Any], student: nn.Module) -> dict[str, torch.Tensor]:
        z, mask, tokens = _require_trajectory(batch)
        depth = z.shape[1]
        a, b = _sample_pairs(z)
        z_a = _gather_depth(z, a)
        z_b = _gather_depth(z, b)
        t_a = _normalized_time(a, depth, z.dtype)
        t_b = _normalized_time(b, depth, z.dtype)
        delta = (t_b - t_a).clamp_min(1e-6)

        out = student(z_a, t_a, delta, tokens=tokens, attention_mask=mask, mode="flow_map")
        if out.z_next is None:
            raise ValueError("LagrangianFlowMapLoss requires student output z_next.")
        loss_map = masked_mse(out.z_next, z_b, mask)

        loss_velocity = z.new_zeros(())
        if self.velocity_weight:
            target_v = (z_b - z_a) / delta.reshape(-1, 1, 1)
            loss_velocity = masked_mse(out.velocity, target_v, mask)

        loss_stability = z.new_zeros(())
        if self.stability_weight:
            final_out = student(
                z[:, -1],
                torch.ones(z.shape[0], device=z.device, dtype=z.dtype),
                torch.zeros(z.shape[0], device=z.device, dtype=z.dtype),
                tokens=tokens,
                attention_mask=mask,
                mode="flow_map",
            )
            if final_out.z_next is not None:
                loss_stability = masked_mse(final_out.z_next, z[:, -1], mask)

        total = (
            self.map_weight * loss_map
            + self.velocity_weight * loss_velocity
            + self.stability_weight * loss_stability
        )
        metrics = {
            "loss": total,
            "loss_flow_map": loss_map.detach(),
            "loss_velocity_secant": loss_velocity.detach(),
            "loss_stability": loss_stability.detach(),
            "metric_interval": delta.mean().detach(),
        }
        metrics.update(_metric_defaults(batch, z))
        return metrics

    forward = compute


class CompositionalFlowMapLoss(nn.Module):
    """Teacher-anchored shortcut/PSD-style consistency over latent flow maps."""

    def __init__(
        self,
        supervised_weight: float = 1.0,
        consistency_weight: float = 1.0,
        midpoint_weight: float = 0.5,
        detach_composed_target: bool = True,
    ):
        super().__init__()
        self.supervised_weight = float(supervised_weight)
        self.consistency_weight = float(consistency_weight)
        self.midpoint_weight = float(midpoint_weight)
        self.detach_composed_target = bool(detach_composed_target)

    def compute(self, batch: dict[str, Any], student: nn.Module) -> dict[str, torch.Tensor]:
        z, mask, tokens = _require_trajectory(batch)
        depth = z.shape[1]
        a, m, b = _sample_triplets(z)
        z_a = _gather_depth(z, a)
        z_m = _gather_depth(z, m)
        z_b = _gather_depth(z, b)
        t_a = _normalized_time(a, depth, z.dtype)
        t_m = _normalized_time(m, depth, z.dtype)
        t_b = _normalized_time(b, depth, z.dtype)
        delta_am = (t_m - t_a).clamp_min(1e-6)
        delta_mb = (t_b - t_m).clamp_min(1e-6)
        delta_ab = (t_b - t_a).clamp_min(1e-6)

        direct = student(z_a, t_a, delta_ab, tokens=tokens, attention_mask=mask, mode="flow_map").z_next
        first = student(z_a, t_a, delta_am, tokens=tokens, attention_mask=mask, mode="flow_map").z_next
        if direct is None or first is None:
            raise ValueError("CompositionalFlowMapLoss requires student output z_next.")
        second_input = first.detach() if self.detach_composed_target else first
        composed = student(
            second_input,
            t_m,
            delta_mb,
            tokens=tokens,
            attention_mask=mask,
            mode="flow_map",
        ).z_next
        if composed is None:
            raise ValueError("CompositionalFlowMapLoss requires student output z_next.")
        composed_target = composed.detach() if self.detach_composed_target else composed

        loss_supervised = masked_mse(direct, z_b, mask)
        loss_consistency = masked_mse(direct, composed_target, mask)
        loss_midpoint = masked_mse(first, z_m, mask)
        total = (
            self.supervised_weight * loss_supervised
            + self.consistency_weight * loss_consistency
            + self.midpoint_weight * loss_midpoint
        )
        metrics = {
            "loss": total,
            "loss_flow_map_supervised": loss_supervised.detach(),
            "loss_flow_map_composition": loss_consistency.detach(),
            "loss_flow_map_midpoint": loss_midpoint.detach(),
            "metric_interval": delta_ab.mean().detach(),
            "metric_midpoint_interval": delta_am.mean().detach(),
        }
        metrics.update(_metric_defaults(batch, z))
        return metrics

    forward = compute


class EulerianMeanFlowTrajectoryLoss(nn.Module):
    """MeanFlow-style average-velocity supervision on teacher latent trajectory pairs."""

    def __init__(
        self,
        avg_velocity_weight: float = 1.0,
        velocity_weight: float = 0.0,
        jvp_weight: float = 0.0,
        norm_eps: float = 0.01,
        norm_p: float = 1.0,
        jvp_autocast_enabled: bool = False,
        jvp_fallback: str = "finite_difference",
        finite_difference_eps: float = 1e-3,
    ):
        super().__init__()
        self.avg_velocity_weight = float(avg_velocity_weight)
        self.velocity_weight = float(velocity_weight)
        self.jvp_weight = float(jvp_weight)
        self.norm_eps = float(norm_eps)
        self.norm_p = float(norm_p)
        self.jvp_autocast_enabled = bool(jvp_autocast_enabled)
        self.jvp_fallback = str(jvp_fallback)
        self.finite_difference_eps = float(finite_difference_eps)

    def compute(self, batch: dict[str, Any], student: nn.Module) -> dict[str, torch.Tensor]:
        z, mask, tokens = _require_trajectory(batch)
        depth = z.shape[1]
        a, b = _sample_pairs(z)
        z_a = _gather_depth(z, a)
        z_b = _gather_depth(z, b)
        t_a = _normalized_time(a, depth, z.dtype)
        t_b = _normalized_time(b, depth, z.dtype)
        delta = (t_b - t_a).clamp_min(1e-6)
        target_avg = (z_b - z_a) / delta.reshape(-1, 1, 1)

        out = student(z_a, t_a, delta, tokens=tokens, attention_mask=mask, mode="flow_map")
        if out.avg_velocity is None:
            raise ValueError("EulerianMeanFlowTrajectoryLoss requires student output avg_velocity.")
        avg_error = out.avg_velocity - target_avg.detach()
        loss_avg = masked_mse(out.avg_velocity, target_avg.detach(), mask)
        adaptive = (loss_avg.detach() + self.norm_eps) ** self.norm_p
        loss_avg = loss_avg / adaptive

        loss_velocity = z.new_zeros(())
        if self.velocity_weight:
            loss_velocity = masked_mse(out.velocity, target_avg.detach(), mask)

        loss_jvp = z.new_zeros(())
        if self.jvp_weight:
            loss_jvp = self._jvp_loss(student, z_a, t_a, delta, target_avg, tokens, mask)

        total = (
            self.avg_velocity_weight * loss_avg
            + self.velocity_weight * loss_velocity
            + self.jvp_weight * loss_jvp
        )
        metrics = {
            "loss": total,
            "loss_meanflow_avg_velocity": loss_avg.detach(),
            "loss_velocity_secant": loss_velocity.detach(),
            "loss_meanflow_jvp": loss_jvp.detach(),
            "metric_meanflow_interval": delta.mean().detach(),
            "metric_meanflow_avg_error": avg_error.pow(2).mean().detach(),
        }
        metrics.update(_metric_defaults(batch, z))
        return metrics

    def _jvp_loss(
        self,
        student: nn.Module,
        z_a: torch.Tensor,
        t_a: torch.Tensor,
        delta: torch.Tensor,
        target_avg: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        def avg_velocity_fn(z_in: torch.Tensor, t_in: torch.Tensor, delta_in: torch.Tensor) -> torch.Tensor:
            out = student(z_in, t_in, delta_in, tokens=tokens, attention_mask=mask, mode="flow_map")
            if out.avg_velocity is None:
                raise ValueError("EulerianMeanFlowTrajectoryLoss requires student output avg_velocity.")
            return out.avg_velocity

        autocast_ctx = torch.amp.autocast("cuda", enabled=False) if z_a.is_cuda and not self.jvp_autocast_enabled else nullcontext()
        with autocast_ctx:
            try:
                _, du = torch.func.jvp(
                    avg_velocity_fn,
                    (z_a, t_a, delta),
                    (target_avg.detach(), torch.ones_like(t_a), torch.zeros_like(delta)),
                )
            except (RuntimeError, NotImplementedError):
                if self.jvp_fallback != "finite_difference":
                    raise
                eps = self.finite_difference_eps
                base = avg_velocity_fn(z_a, t_a, delta)
                shifted = avg_velocity_fn(z_a + eps * target_avg.detach(), t_a + eps, delta)
                du = (shifted - base) / eps
        meanflow_estimate = avg_velocity_fn(z_a, t_a, delta) + delta.reshape(-1, 1, 1) * du.detach()
        return masked_mse(meanflow_estimate, target_avg.detach(), mask)

    forward = compute
