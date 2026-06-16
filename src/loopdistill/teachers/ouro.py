from __future__ import annotations

import os
from contextlib import nullcontext

import torch

from loopdistill.teachers.base import TeacherOutput, TeacherRunner


class OuroTeacher(TeacherRunner):
    """HuggingFace Ouro adapter.

    This adapter captures hidden states from repeated forward passes with `total_ut_steps`.
    Exact loop-state hooks can be added once the loaded Ouro implementation exposes stable
    internal module names.
    """

    def __init__(
        self,
        model_id: str = "ByteDance/Ouro-1.4B",
        teacher_id: str = "ouro",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        attn_implementation: str | None = "flash_attention_2",
        total_ut_steps: int = 4,
    ):
        self.model_id = model_id
        self.teacher_id = teacher_id
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.attn_implementation = attn_implementation
        self.total_ut_steps = total_ut_steps
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if "HF_TOKEN" not in os.environ:
            raise RuntimeError("OuroTeacher requires HF_TOKEN in the environment.")
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(self.model_id, token=os.environ["HF_TOKEN"], trust_remote_code=True)
        if hasattr(config, "total_ut_steps"):
            config.total_ut_steps = self.total_ut_steps
        kwargs = {
            "config": config,
            "token": os.environ["HF_TOKEN"],
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
            "output_hidden_states": True,
        }
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).eval()
        return self._model

    @torch.no_grad()
    def run_batch(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        depths: list[int],
    ) -> TeacherOutput:
        model = self._load_model()
        device = next(model.parameters()).device
        tokens = tokens.to(device)
        attention_mask = attention_mask.to(device)
        max_depth = max(depths) if depths else self.total_ut_steps
        states = []
        logits = []
        for depth in range(max_depth + 1):
            if hasattr(model.config, "total_ut_steps"):
                model.config.total_ut_steps = max(depth, 1)
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=False) if tokens.is_cuda else nullcontext()
            with ctx:
                out = model(input_ids=tokens, attention_mask=attention_mask, output_hidden_states=True)
            states.append(out.hidden_states[-1].float())
            logits.append(out.logits.float())
        z = torch.stack(states, dim=1)
        logits_t = torch.stack(logits, dim=1)
        batch = tokens.shape[0]
        return TeacherOutput(
            z=z.cpu(),
            logits=logits_t.cpu(),
            loss_K=torch.zeros(batch),
            residual_norm=(z[:, -1] - z[:, -2]).pow(2).mean(dim=(1, 2)).sqrt().cpu()
            if z.shape[1] > 1
            else torch.zeros(batch),
            solver_iters=torch.full((batch,), max_depth, dtype=torch.float32),
        )
