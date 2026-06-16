from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class TeacherOutput:
    z: torch.Tensor
    logits: torch.Tensor | None
    loss_K: torch.Tensor
    residual_norm: torch.Tensor
    solver_iters: torch.Tensor


class TeacherRunner(ABC):
    teacher_id: str

    @abstractmethod
    def run_batch(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        depths: list[int],
    ) -> TeacherOutput:
        raise NotImplementedError

    def project_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent states through the teacher endpoint LM head without running the loop."""
        raise NotImplementedError(f"{type(self).__name__} does not expose teacher-head projection.")
