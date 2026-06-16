from __future__ import annotations

import torch

from loopdistill.teachers.base import TeacherOutput, TeacherRunner


class MockTeacher(TeacherRunner):
    """Deterministic synthetic loop trajectory generator for local tests."""

    def __init__(
        self,
        teacher_id: str = "mock",
        latent_dim: int = 32,
        vocab_size: int = 128,
        max_depth: int = 4,
        noise_scale: float = 0.05,
    ):
        self.teacher_id = teacher_id
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_depth = max_depth
        self.noise_scale = noise_scale

    def _projection(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        proj = torch.arange(self.latent_dim * self.vocab_size, device=device, dtype=dtype).reshape(
            self.latent_dim, self.vocab_size
        )
        return torch.sin(proj / proj.numel())

    def project_logits(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bld,dv->blv", z, self._projection(z.device, z.dtype)).float()

    def run_batch(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        depths: list[int],
    ) -> TeacherOutput:
        device = tokens.device
        batch, seq_len = tokens.shape
        K = max(depths) if depths else self.max_depth
        base = torch.sin(tokens.float().unsqueeze(-1) / (torch.arange(self.latent_dim, device=device) + 1))
        base = base * attention_mask.unsqueeze(-1).float()
        fixed = torch.tanh(base)
        states = []
        for k in range(K + 1):
            alpha = k / max(K, 1)
            noise = self.noise_scale * (1.0 - alpha) * torch.cos(base + k)
            states.append((1 - alpha) * base + alpha * fixed + noise)
        z = torch.stack(states, dim=1)
        logits = torch.einsum("bkld,dv->bklv", z, self._projection(device, z.dtype))
        residual_norm = (z[:, 1:] - z[:, :-1]).pow(2).mean(dim=(2, 3)).sqrt()
        return TeacherOutput(
            z=z,
            logits=logits,
            loss_K=torch.zeros(batch, device=device),
            residual_norm=residual_norm[:, -1],
            solver_iters=torch.full((batch,), K, dtype=torch.float32, device=device),
        )
