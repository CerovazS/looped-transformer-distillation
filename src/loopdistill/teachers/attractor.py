from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

from loopdistill.teachers.base import TeacherOutput, TeacherRunner


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype!r}")
    return mapping[dtype]


class AttractorTeacher(TeacherRunner):
    """Read-only adapter for Solve The Loop Attractor/EQLM checkpoints."""

    def __init__(
        self,
        repo_path: str,
        checkpoint_dir: str,
        teacher_id: str = "attractor",
        device: str = "auto",
        dtype: str | None = "auto",
        storage_dtype: str | None = "bfloat16",
        max_depth: int = 8,
        return_logits: bool = False,
        logit_depths: str = "final",
        attn_impl: str | None = "sdpa",
        strict_load: bool = False,
        **config_overrides: Any,
    ):
        self.teacher_id = teacher_id
        self.repo_path = str(Path(repo_path).expanduser())
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype)
        self.storage_dtype = _resolve_dtype(storage_dtype)
        self.max_depth = int(max_depth)
        self.return_logits = bool(return_logits)
        self.logit_depths = logit_depths
        self.attn_impl = attn_impl
        self.strict_load = strict_load
        self.config_overrides = config_overrides
        self._model = None
        self._tokenizer = None
        self.vocab_size: int | None = None
        self.latent_dim: int | None = None

    @property
    def tokenizer_id(self) -> str:
        return str(self.checkpoint_dir)

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
            freqs_cis = model.freqs_cis[:, : tokens.shape[1]]
            context = model._encode(tokens, freqs_cis, attention_mask)
            states = []
            final_info = {"iters": 0, "rel_residual": 0.0}
            for depth in depths:
                if depth == 0:
                    state = context
                else:
                    state, final_info = model._solve_forward(
                        context,
                        freqs_cis,
                        attention_mask,
                        max_iter_override=depth,
                    )
                states.append(state.detach().to(dtype=self.storage_dtype))
            z = torch.stack(states, dim=1)
            logits = self._maybe_project_logits(model, states)

        return TeacherOutput(
            z=z,
            logits=logits,
            loss_K=torch.zeros(batch, device=self.device),
            residual_norm=torch.full(
                (batch,),
                float(final_info.get("rel_residual", 0.0)),
                dtype=torch.float32,
                device=self.device,
            ),
            solver_iters=torch.full(
                (batch,),
                float(final_info.get("iters", max(depths))),
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def _validate_depths(self, depths: list[int]) -> list[int]:
        if not depths:
            depths = list(range(self.max_depth + 1))
        depths = [int(depth) for depth in depths]
        expected = list(range(max(depths) + 1))
        if depths != expected:
            raise ValueError(
                "P0 Attractor extraction expects consecutive depths "
                f"{expected}, got {depths}."
            )
        return depths

    def _ensure_repo_importable(self) -> None:
        repo = Path(self.repo_path).resolve()
        if not repo.exists():
            raise FileNotFoundError(f"Attractor repo_path not found: {repo}")
        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        tokenizer_path = self.checkpoint_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.checkpoint_dir),
                add_bos_token=False,
                add_eos_token=False,
            )
        except Exception:
            self._tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_path),
                add_bos_token=False,
                add_eos_token=False,
            )
        self.vocab_size = int(self._tokenizer.vocab_size)
        return self._tokenizer

    def _load_model(self):
        if self._model is not None:
            return self._model
        self._ensure_repo_importable()
        config = self._load_config()
        model = config.construct_model()
        weights_path = self._find_weights()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        cleaned = {}
        for key, value in state_dict.items():
            for prefix in ("module.", "_orig_mod.", "model.", "_forward_module."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            cleaned[key] = value
        model.load_state_dict(cleaned, strict=self.strict_load)
        if self.dtype is not None:
            model = model.to(dtype=self.dtype)
        model = model.to(device=self.device)
        model.eval()
        self._model = model
        self.vocab_size = int(model.config.padded_vocab_size)
        self.latent_dim = int(model.config.n_embd)
        return model

    def _load_config(self):
        self._ensure_repo_importable()
        from attractor.models.config import RoPESettings

        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Attractor config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
        class_name = config_dict.get("_class_name", "EQLMConfig")
        if class_name == "EQLMConfig":
            from attractor.models.eqlm.config import EQLMConfig as config_cls
        elif class_name == "AttractorConfig":
            from attractor.models.attractor.config import AttractorConfig as config_cls
        else:
            raise ValueError(f"Unsupported Attractor config class: {class_name}")

        if "rope_settings" in config_dict and isinstance(config_dict["rope_settings"], dict):
            config_dict["rope_settings"] = RoPESettings(**config_dict["rope_settings"])
        for key in ("_class_name", "init"):
            config_dict.pop(key, None)
        if self.attn_impl is not None:
            config_dict["attn_impl"] = self.attn_impl
        config_dict.update(self.config_overrides)
        return config_cls(**config_dict)

    def _find_weights(self) -> Path:
        for name in ("pytorch_model.bin", "model.safetensors", "model.bin", "model.pt"):
            path = self.checkpoint_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(f"No supported Attractor weights found in {self.checkpoint_dir}")

    def project_logits(self, z: torch.Tensor) -> torch.Tensor:
        model = self._load_model()
        states = z.to(device=self.device)
        if self.dtype is not None:
            states = states.to(dtype=self.dtype)
        with torch.no_grad():
            logits = self._project_logits(model, states)
        return logits

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
            raise ValueError(f"Expected latent states with shape [B,L,D] or [B,K,L,D], got {tuple(states.shape)}.")
        batch, depth, seq_len, dim = states.shape
        flat_states = states.reshape(batch * depth, seq_len, dim)
        x = model.transformer.ln_f(flat_states)
        if model.config.use_fused_head == "full-triton":
            weight = model.lm_head.weight
            out = torch.matmul(x, weight.T if model.config.tie_embeddings else weight)
        else:
            out = model.lm_head(x)
        out = out.float() * model.config.init.logit_scale
        if model.config.logit_softcap is not None:
            softcap = model.config.logit_softcap
            out = softcap * torch.tanh(out / softcap)
        out = out.reshape(batch, depth, seq_len, out.shape[-1])
        return out[:, 0] if squeeze_depth else out
