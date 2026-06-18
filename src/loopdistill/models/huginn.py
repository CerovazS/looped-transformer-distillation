from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from loopdistill.models.student import StudentOutput
from loopdistill.teachers.attractor import _resolve_dtype


class HuginnRecurrentStudent(nn.Module):
    """Trainable Huginn recurrent transition block for K>S loop compression."""

    def __init__(
        self,
        model_id: str = "tomg-group-umd/huginn-0125",
        checkpoint_dir: str | None = None,
        dtype: str | torch.dtype | None = "bfloat16",
        trust_remote_code: bool = True,
        target_depth: int = 16,
        trainable_parts: Sequence[str] = ("adapter", "core_block"),
        return_logits: bool = False,
        share_base_model_from_teacher: bool = False,
        **model_kwargs,
    ):
        super().__init__()
        self.model_id = model_id
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir).expanduser()
        self.dtype = _resolve_dtype(dtype)
        self.trust_remote_code = bool(trust_remote_code)
        self.target_depth = int(target_depth)
        self.trainable_parts = tuple(str(part) for part in trainable_parts)
        self.return_logits = bool(return_logits)
        self.share_base_model_from_teacher = bool(share_base_model_from_teacher)
        self.model_kwargs = model_kwargs
        if self.target_depth <= 0:
            raise ValueError("target_depth must be positive.")
        object.__setattr__(self, "_shared_model", None)
        if self.share_base_model_from_teacher:
            self.model = None
        else:
            self.model = self._load_model()
            self._configure_trainable_parameters()

    @property
    def pretrained_name_or_path(self) -> str:
        return str(self.checkpoint_dir or self.model_id)

    def initialize_from_teacher(self, teacher) -> None:
        if not self.share_base_model_from_teacher or self.model is not None:
            return
        if not hasattr(teacher, "_load_model"):
            raise ValueError("share_base_model_from_teacher requires a HuginnTeacher-like object.")
        base_model = teacher._load_model()
        object.__setattr__(self, "_shared_model", base_model)
        base_modules = self._module_map(base_model)
        unknown = sorted(set(self.trainable_parts) - set(base_modules))
        if unknown:
            raise ValueError(
                f"Unknown Huginn trainable_parts: {unknown}. "
                f"Expected one of {sorted(base_modules)} or 'all'."
            )
        selected = tuple(base_modules) if "all" in self.trainable_parts else self.trainable_parts
        for part in selected:
            module_copy = copy.deepcopy(base_modules[part])
            for param in module_copy.parameters():
                param.requires_grad = True
            setattr(self, part, module_copy)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        delta: torch.Tensor,
        tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        mode: str = "flow_map",
    ) -> StudentOutput:
        if tokens is None:
            raise ValueError("HuginnRecurrentStudent requires token ids to rebuild Huginn input embeddings.")
        if z_t.dim() != 3:
            raise ValueError(f"Expected z_t with shape [B,L,D], got {tuple(z_t.shape)}.")
        model = self._active_model()
        tokens = tokens.to(device=z_t.device, dtype=torch.long)
        with torch.no_grad():
            input_embeds, _ = model.embed_inputs(tokens, attention_mask=None)
        input_embeds = input_embeds.to(device=z_t.device, dtype=z_t.dtype)
        with self._using_trainable_modules(model):
            next_z = self._grouped_core_forward(model, z_t, input_embeds, t)
        safe_delta = delta.reshape(-1, 1, 1).to(device=z_t.device, dtype=z_t.dtype).clamp_min(1e-6)
        velocity = (next_z - z_t) / safe_delta
        logits = None
        if self.return_logits:
            logits = model.predict_from_latents(next_z).logits
        return StudentOutput(velocity=velocity, z_next=next_z, avg_velocity=velocity, logits=logits)

    def _grouped_core_forward(
        self,
        model: nn.Module,
        z_t: torch.Tensor,
        input_embeds: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        current_steps = self._current_steps(t, z_t.device)
        out = torch.empty_like(z_t)
        for step in torch.unique(current_steps).tolist():
            selector = current_steps == int(step)
            block_idx = self._block_idx_before_recurrence(model, int(step), z_t.device)
            freqs_cis = model.freqs_cis[:, : z_t.shape[1]].to(device=z_t.device)
            next_z, _ = model.core_block_forward(
                z_t[selector],
                input_embeds[selector],
                freqs_cis,
                None,
                None,
                block_idx,
                current_step=int(step),
            )
            out[selector] = next_z
        return out

    def _current_steps(self, t: torch.Tensor, device: torch.device) -> torch.Tensor:
        steps = torch.round(t.float().to(device) * self.target_depth).long()
        return steps.clamp(min=0, max=max(self.target_depth - 1, 0))

    def _block_idx_before_recurrence(
        self,
        model: nn.Module,
        current_step: int,
        device: torch.device,
    ) -> torch.Tensor:
        config = model.config
        block_idx = int(config.n_layers_in_prelude) - 1
        block_idx += int(current_step) * int(config.n_layers_in_recurrent_block)
        return torch.tensor(block_idx, device=device, dtype=torch.long)

    def _active_model(self) -> nn.Module:
        shared_model = getattr(self, "_shared_model", None)
        if shared_model is not None:
            return shared_model
        if self.model is None:
            raise RuntimeError(
                "HuginnRecurrentStudent has no loaded model. "
                "Call initialize_from_teacher() or disable share_base_model_from_teacher."
            )
        return self.model

    @contextmanager
    def _using_trainable_modules(self, model: nn.Module):
        if not getattr(self, "share_base_model_from_teacher", False):
            yield
            return
        transformer = model.transformer
        originals = {}
        selected = tuple(self._module_map(model)) if "all" in self.trainable_parts else self.trainable_parts
        for part in selected:
            if part not in {"adapter", "core_block"}:
                raise ValueError(
                    "Shared Huginn mode currently supports trainable_parts "
                    "['adapter', 'core_block']; got "
                    f"{self.trainable_parts}."
                )
            if not hasattr(self, part):
                raise RuntimeError(f"Shared Huginn student has not initialized {part!r}.")
            originals[part] = getattr(transformer, part)
            setattr(transformer, part, getattr(self, part))
        try:
            yield
        finally:
            for part, module in originals.items():
                setattr(transformer, part, module)

    def _load_model(self) -> nn.Module:
        from transformers import AutoModelForCausalLM

        load_kwargs = dict(self.model_kwargs)
        if self.dtype is not None:
            load_kwargs.setdefault("torch_dtype", self.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_name_or_path,
            trust_remote_code=self.trust_remote_code,
            **load_kwargs,
        )
        return model

    def _configure_trainable_parameters(self) -> None:
        if "all" in self.trainable_parts:
            for param in self.model.parameters():
                param.requires_grad = True
            return
        for param in self.model.parameters():
            param.requires_grad = False
        modules = self._module_map(self.model)
        unknown = sorted(set(self.trainable_parts) - set(modules))
        if unknown:
            raise ValueError(f"Unknown Huginn trainable_parts: {unknown}. Expected one of {sorted(modules)} or 'all'.")
        for part in self.trainable_parts:
            for param in modules[part].parameters():
                param.requires_grad = True

    def _module_map(self, model: nn.Module) -> dict[str, nn.Module]:
        return {
            "wte": model.transformer.wte,
            "prelude": model.transformer.prelude,
            "adapter": model.transformer.adapter,
            "core_block": model.transformer.core_block,
            "coda": model.transformer.coda,
            "ln_f": model.transformer.ln_f,
            "lm_head": model.lm_head,
        }
