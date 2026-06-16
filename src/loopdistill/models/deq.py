from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class DEQLatentLoop(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        core: str = "sliced",
        f_solver: str = "fixed_point_iter",
        b_solver: str = "fixed_point_iter",
        f_max_iter: int = 16,
        b_max_iter: int = 16,
        f_tol: float = 1e-3,
        b_tol: float = 1e-6,
        tau: float = 1.0,
        ift: bool = False,
        hook_ift: bool = False,
    ):
        super().__init__()
        self.block = block
        try:
            from torchdeq import get_deq
        except ImportError as exc:
            raise ImportError("DEQLatentLoop requires torchdeq. Install with `uv add torchdeq`.") from exc
        self.deq = get_deq(
            {
                "core": core,
                "f_solver": f_solver,
                "b_solver": b_solver,
                "f_max_iter": f_max_iter,
                "b_max_iter": b_max_iter,
                "f_tol": f_tol,
                "b_tol": b_tol,
                "tau": tau,
                "ift": ift,
                "hook_ift": hook_ift,
            }
        )

    def forward(
        self,
        z0: torch.Tensor,
        func: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **solver_kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        f = func or self.block
        z_out, info = self.deq(f, z0, solver_kwargs=solver_kwargs)
        z_star = z_out[-1]
        if isinstance(z_star, (list, tuple)):
            z_star = z_star[0]
        return z_star, info
