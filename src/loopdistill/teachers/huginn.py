from __future__ import annotations

from pathlib import Path

import torch

from loopdistill.teachers.attractor import _resolve_device, _resolve_dtype
from loopdistill.teachers.base import TeacherOutput, TeacherRunner


class HuginnTeacher(TeacherRunner):
    """Adapter for Huginn-0125 recurrent-depth language-model checkpoints."""

    def __init__(
        self,
        model_id: str = "tomg-group-umd/huginn-0125",
        checkpoint_dir: str | None = None,
        teacher_id: str = "huginn",
        device: str = "auto",
        dtype: str | None = "auto",
        storage_dtype: str | None = "bfloat16",
        max_depth: int = 32,
        return_logits: bool = False,
        logit_depths: str = "final",
        trust_remote_code: bool = True,
        init_scale: float = 1.0,
        **model_kwargs,
    ):
        self.teacher_id = teacher_id
        self.model_id = model_id
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir).expanduser()
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype)
        self.storage_dtype = _resolve_dtype(storage_dtype)
        self.max_depth = int(max_depth)
        self.return_logits = bool(return_logits)
        self.logit_depths = logit_depths
        self.trust_remote_code = bool(trust_remote_code)
        self.init_scale = float(init_scale)
        self.model_kwargs = model_kwargs
        self._model = None
        self._tokenizer = None
        self.vocab_size: int | None = None
        self.latent_dim: int | None = None

    @property
    def tokenizer_id(self) -> str:
        return str(self.checkpoint_dir or self.model_id)

    @property
    def pretrained_name_or_path(self) -> str:
        return str(self.checkpoint_dir or self.model_id)

    def set_device(self, device: str | torch.device) -> None:
        self.device = _resolve_device(device)
        if self._model is not None:
            self._model = self._model.to(device=self.device)

    def encode_text(self, text: str) -> list[int]:
        tokenizer = self._load_tokenizer()
        return tokenizer.encode(text, add_special_tokens=False)

    def run_batch(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        depths: list[int],
    ) -> TeacherOutput:
        model = self._load_model()
        depths = self._validate_depths(depths)
        tokens = tokens.to(self.device, dtype=torch.long)
        attention_mask = attention_mask.to(self.device)
        batch = tokens.shape[0]

        with torch.no_grad():
            # Huginn's public forward path ignores the 2D attention mask and uses no BlockMask by
            # default; passing a 2D mask to embed_inputs triggers its lower-level 3D mask branch.
            input_embeds, block_idx = model.embed_inputs(tokens, attention_mask=None)
            state = model.initialize_state(input_embeds, scale=self.init_scale)
            states = [state.detach()]
            previous_state = None
            for step in range(max(depths)):
                previous_state = state
                state, block_idx, _ = model.iterate_one_step(
                    input_embeds,
                    state,
                    block_idx=block_idx,
                    attention_mask=None,
                    past_key_values=None,
                    current_step=step,
                )
                states.append(state.detach())

            selected_states = [states[depth].to(dtype=self.storage_dtype) for depth in depths]
            z = torch.stack(selected_states, dim=1)
            logits = self._maybe_project_logits(model, selected_states)

        residual = torch.zeros(batch, dtype=torch.float32, device=self.device)
        if previous_state is not None and depths[-1] > 0:
            denom = state.float().norm(dim=-1).clamp_min(1e-8)
            residual = (state.float() - previous_state.float()).norm(dim=-1) / denom
            residual = residual.mean(dim=1)

        return TeacherOutput(
            z=z,
            logits=logits,
            loss_K=torch.zeros(batch, device=self.device),
            residual_norm=residual,
            solver_iters=torch.full(
                (batch,),
                float(max(depths) if depths else self.max_depth),
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def project_logits(self, z: torch.Tensor) -> torch.Tensor:
        model = self._load_model()
        states = z.to(device=self.device)
        if self.dtype is not None:
            states = states.to(dtype=self.dtype)
        with torch.no_grad():
            return self._project_logits(model, states)

    def _validate_depths(self, depths: list[int]) -> list[int]:
        if not depths:
            depths = list(range(self.max_depth + 1))
        depths = [int(depth) for depth in depths]
        expected = list(range(max(depths) + 1))
        if depths != expected:
            raise ValueError(
                "Huginn extraction expects consecutive depths "
                f"{expected}, got {depths}."
            )
        return depths

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.vocab_size = int(self._tokenizer.vocab_size)
        return self._tokenizer

    def _load_model(self):
        if self._model is not None:
            return self._model
        from transformers import AutoModelForCausalLM

        load_kwargs = dict(self.model_kwargs)
        if self.dtype is not None:
            load_kwargs.setdefault("torch_dtype", self.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_name_or_path,
            trust_remote_code=self.trust_remote_code,
            **load_kwargs,
        )
        model = model.to(device=self.device)
        model.eval()
        self._model = model
        self.vocab_size = int(getattr(model.config, "padded_vocab_size", model.config.vocab_size))
        self.latent_dim = int(model.config.n_embd)
        return model

    def _maybe_project_logits(self, model, states: list[torch.Tensor]) -> torch.Tensor | None:
        if not self.return_logits:
            return None
        selected = states if self.logit_depths == "all" else [states[-1]]
        return self._project_logits(model, torch.stack(selected, dim=1)).detach()

    def _project_logits(self, model, states: torch.Tensor) -> torch.Tensor:
        squeeze_depth = False
        if states.dim() == 3:
            states = states.unsqueeze(1)
            squeeze_depth = True
        if states.dim() != 4:
            raise ValueError(
                "Expected latent states with shape [B,L,D] or [B,K,L,D], "
                f"got {tuple(states.shape)}."
            )
        batch, depth, seq_len, dim = states.shape
        flat_states = states.reshape(batch * depth, seq_len, dim)
        outputs = model.predict_from_latents(flat_states)
        logits = outputs.logits.reshape(batch, depth, seq_len, -1)
        return logits[:, 0] if squeeze_depth else logits
