from __future__ import annotations

import torch
from torch import nn

from loopdistill.losses.meanflow import MeanFlowLoss
from loopdistill.losses.p0 import LoopDistillationLoss
from loopdistill.losses.shortcut import ShortcutConsistencyLoss
from loopdistill.models.student import StudentFlowModel


def _batch():
    batch, depth, seq_len, dim, vocab = 3, 5, 7, 8, 11
    z0 = torch.randn(batch, seq_len, dim)
    direction = torch.randn(batch, seq_len, dim) * 0.1
    z = torch.stack([z0 + k * direction for k in range(depth)], dim=1)
    return {
        "tokens": torch.randint(0, vocab, (batch, seq_len)),
        "attention_mask": torch.ones(batch, seq_len, dtype=torch.bool),
        "teacher_id": ["mock"] * batch,
        "K": torch.full((batch,), depth - 1),
        "z": z,
        "logits": torch.randn(batch, depth, seq_len, vocab),
        "loss_K": torch.zeros(batch),
        "residual_norm": torch.zeros(batch),
        "solver_iters": torch.full((batch,), depth - 1.0),
    }


def test_p0_loss_backward():
    batch = _batch()
    student = StudentFlowModel(latent_dim=8, hidden_dim=32, num_layers=1, num_heads=4, vocab_size=11)
    loss_module = LoopDistillationLoss()
    metrics = loss_module.compute(batch, student)
    metrics["loss"].backward()
    assert torch.isfinite(metrics["loss"])
    assert any(param.grad is not None for param in student.parameters())


def test_meanflow_loss_backward():
    model = nn.Sequential(nn.Linear(8 + 2, 16), nn.SiLU(), nn.Linear(16, 8))

    def fn(z, t, delta):
        cond = torch.stack([t, delta], dim=-1).unsqueeze(1).expand(z.shape[0], z.shape[1], 2)
        return model(torch.cat([z, cond], dim=-1))

    loss = MeanFlowLoss()(torch.randn(2, 4, 8), fn)["loss_meanflow"]
    loss.backward()
    assert torch.isfinite(loss)


def test_shortcut_linear_map_zeroish():
    class LinearMap(nn.Module):
        def forward(self, z_t, t, delta, **kwargs):
            from loopdistill.models.student import StudentOutput

            return StudentOutput(velocity=z_t, z_next=z_t + delta.reshape(-1, 1, 1))

    z = torch.randn(2, 4, 8)
    t = torch.zeros(2)
    delta = torch.ones(2) * 0.5
    out = ShortcutConsistencyLoss()(LinearMap(), z, t, delta)
    assert out["loss_shortcut"] < 1e-6
