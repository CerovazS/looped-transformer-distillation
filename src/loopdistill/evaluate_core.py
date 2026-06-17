from __future__ import annotations

import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from loopdistill.teachers.base import TeacherRunner
from loopdistill.utils.logging import info, ok, warn
from loopdistill.utils.run import ensure_run_dirs, save_resolved_config, write_run_summary


@dataclass
class LogitBatch:
    teacher_logits: torch.Tensor
    student_logits: torch.Tensor | None = None
    solver_iters: torch.Tensor | None = None
    rel_residual: torch.Tensor | None = None


class ScalarAverager:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0.0

    def add(self, value: torch.Tensor | float, count: torch.Tensor | float) -> None:
        v = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        c = float(count.detach().cpu()) if isinstance(count, torch.Tensor) else float(count)
        self.total += v
        self.count += c

    def mean(self) -> float | None:
        if self.count <= 0:
            return None
        return self.total / self.count


class MetricAccumulators:
    def __init__(self) -> None:
        self.teacher_nll = ScalarAverager()
        self.student_nll = ScalarAverager()
        self.kl = ScalarAverager()
        self.top1_agreement = ScalarAverager()
        self.topk_overlap = ScalarAverager()
        self.teacher_next_token_accuracy = ScalarAverager()
        self.student_next_token_accuracy = ScalarAverager()
        self.solver_iters = ScalarAverager()
        self.rel_residual = ScalarAverager()

    def to_metrics(self, *, max_ppl_exp: float, prefix: str = "") -> dict[str, float]:
        metrics: dict[str, float] = {}
        teacher_nll = self.teacher_nll.mean()
        student_nll = self.student_nll.mean()
        if teacher_nll is not None:
            metrics[f"{prefix}nll_teacher"] = teacher_nll
            metrics[f"{prefix}ppl_teacher"] = math.exp(min(teacher_nll, max_ppl_exp))
        if student_nll is not None:
            metrics[f"{prefix}nll_student"] = student_nll
            metrics[f"{prefix}ppl_student"] = math.exp(min(student_nll, max_ppl_exp))
        if teacher_nll is not None and student_nll is not None:
            metrics[f"{prefix}nll_delta"] = student_nll - teacher_nll
            metrics[f"{prefix}ppl_delta"] = math.exp(min(student_nll, max_ppl_exp)) - math.exp(
                min(teacher_nll, max_ppl_exp)
            )
        for name, avg in (
            ("kl_student_teacher", self.kl),
            ("top1_agreement", self.top1_agreement),
            ("topk_overlap", self.topk_overlap),
            ("teacher_next_token_accuracy", self.teacher_next_token_accuracy),
            ("student_next_token_accuracy", self.student_next_token_accuracy),
            ("solver_iters", self.solver_iters),
            ("rel_residual", self.rel_residual),
        ):
            value = avg.mean()
            if value is not None:
                metrics[f"{prefix}{name}"] = value
        return metrics


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str | None) -> torch.dtype | None:
    if dtype is None or dtype == "none":
        return None
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


def _as_step(value: Any, *, auto_value: int | None) -> int | None:
    if value is None:
        return auto_value
    if isinstance(value, str) and value.lower() == "auto":
        return auto_value
    return int(value)


def _load_config_from_checkpoint(checkpoint_path: str | None) -> DictConfig | None:
    if not checkpoint_path:
        return None
    ckpt = Path(checkpoint_path).expanduser().resolve()
    candidates = []
    for parent in (ckpt.parent, *ckpt.parents):
        candidates.append(parent / "artifacts" / "config_resolved.yaml")
        candidates.append(parent.parent / "artifacts" / "config_resolved.yaml")
    for path in candidates:
        if path.exists():
            return OmegaConf.load(path)
    return None


def _student_cfg(cfg: DictConfig) -> DictConfig:
    if cfg.eval.student_config_path:
        loaded = OmegaConf.load(str(Path(cfg.eval.student_config_path).expanduser()))
        return loaded.student if "student" in loaded else loaded
    inferred = _load_config_from_checkpoint(cfg.eval.student_checkpoint)
    if inferred is not None and "student" in inferred:
        return inferred.student
    return cfg.student


def _student_auto_steps(cfg: DictConfig) -> int | None:
    if cfg.eval.student_config_path:
        loaded = OmegaConf.load(str(Path(cfg.eval.student_config_path).expanduser()))
    else:
        loaded = _load_config_from_checkpoint(cfg.eval.student_checkpoint)
    if loaded is None:
        return None
    for path in ("eval_quality.rollout_steps", "loss.rollout_steps", "teacher.max_depth"):
        value = OmegaConf.select(loaded, path)
        if value is not None:
            return int(value)
    return None


def _load_student_checkpoint(student: nn.Module, checkpoint_path: str, *, strict: bool) -> None:
    path = Path(checkpoint_path).expanduser()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        warn(f"Falling back to weights_only=False for trusted checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported student checkpoint format: {type(state)!r}")
    cleaned = {}
    for key, value in state.items():
        if key.startswith("student."):
            key = key[len("student.") :]
        elif key.startswith("module.student."):
            key = key[len("module.student.") :]
        elif key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    missing, unexpected = student.load_state_dict(cleaned, strict=strict)
    if missing:
        warn(f"Student checkpoint missing keys: {len(missing)}")
    if unexpected:
        warn(f"Student checkpoint unexpected keys: {len(unexpected)}")


def _load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_eval_bundle_dir(tokenized_dir: Path, cfg: DictConfig) -> Path | None:
    if cfg.eval.eval_bundle_dir:
        path = Path(str(cfg.eval.eval_bundle_dir)).expanduser()
        return path if path.exists() else None
    metadata_path = tokenized_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_dir = metadata.get("source_dir")
        if source_dir and Path(source_dir).exists():
            return Path(source_dir)
    if (tokenized_dir / "core.yaml").exists() and (tokenized_dir / "eval_data").exists():
        return tokenized_dir
    if (tokenized_dir / "eval_bundle" / "core.yaml").exists():
        return tokenized_dir / "eval_bundle"
    return None


def _load_core_tokenizer(teacher: TeacherRunner):
    if hasattr(teacher, "_load_tokenizer"):
        return teacher._load_tokenizer()
    raise TypeError("CORE raw evaluation requires a teacher tokenizer or a pretokenized fallback.")


def _bos_token_id(tokenizer: Any, cfg: DictConfig) -> int | None:
    if not bool(cfg.eval.prepend_bos):
        return None
    for attr in ("get_bos_token_id",):
        method = getattr(tokenizer, attr, None)
        if callable(method):
            value = method()
            if value is not None:
                return int(value)
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return None


def _pad_token_id(tokenizer: Any, cfg: DictConfig) -> int:
    if cfg.eval.pad_token_id is not None:
        return int(cfg.eval.pad_token_id)
    bos = _bos_token_id(tokenizer, cfg)
    if bos is not None:
        return int(bos)
    value = getattr(tokenizer, "pad_token_id", None)
    return int(value) if value is not None else 0


def _encode_prompt(tokenizer: Any, text: str, cfg: DictConfig) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos = _bos_token_id(tokenizer, cfg)
    return ([bos] if bos is not None else []) + [int(token) for token in ids]


def _find_common_length(token_sequences: list[list[int]], *, direction: str) -> int:
    min_len = min(len(seq) for seq in token_sequences)
    if direction == "left":
        indices = range(min_len)
    elif direction == "right":
        indices = range(-1, -min_len - 1, -1)
    else:
        raise ValueError("direction must be 'left' or 'right'.")
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def _render_prompts_mc(
    item: dict[str, Any],
    continuation_delimiter: str,
    fewshot_examples: list[dict[str, Any]],
) -> list[str]:
    prefix = "\n".join(
        f"{example['query']}{continuation_delimiter}{example['choices'][int(example['gold'])]}"
        for example in fewshot_examples
    )
    prefix = f"{prefix}\n" if prefix else ""
    return [f"{prefix}{item['query']}{continuation_delimiter}{choice}" for choice in item["choices"]]


def _render_prompts_schema(
    item: dict[str, Any],
    continuation_delimiter: str,
    fewshot_examples: list[dict[str, Any]],
) -> list[str]:
    prefix = "\n".join(
        f"{example['context_options'][int(example['gold'])]}{continuation_delimiter}{example['continuation']}"
        for example in fewshot_examples
    )
    prefix = f"{prefix}\n" if prefix else ""
    return [
        f"{prefix}{context}{continuation_delimiter}{item['continuation']}"
        for context in item["context_options"]
    ]


def _render_prompts_lm(
    item: dict[str, Any],
    continuation_delimiter: str,
    fewshot_examples: list[dict[str, Any]],
) -> list[str]:
    prefix = "\n".join(
        f"{str(example['context']).strip()}{continuation_delimiter}{example['continuation']}"
        for example in fewshot_examples
    )
    prefix = f"{prefix}\n" if prefix else ""
    prompt_without = f"{prefix}{str(item['context']).strip()}{continuation_delimiter}"
    prompt_with = f"{prefix}{str(item['context']).strip()}{continuation_delimiter}{item['continuation']}"
    return [prompt_without, prompt_with]


def _batch_sequences(
    tokenizer: Any,
    prompts: list[str],
    task_type: str,
    cfg: DictConfig,
) -> tuple[list[list[int]], list[int], list[int]]:
    tokens = [_encode_prompt(tokenizer, prompt, cfg) for prompt in prompts]
    if task_type == "multiple_choice":
        start_idx = _find_common_length(tokens, direction="left")
        return tokens, [start_idx] * len(tokens), [len(seq) for seq in tokens]
    if task_type == "schema":
        suffix_length = _find_common_length(tokens, direction="right")
        end_indices = [len(seq) for seq in tokens]
        start_indices = [end_idx - suffix_length for end_idx in end_indices]
        return tokens, start_indices, end_indices
    if task_type == "language_modeling":
        tokens_without, tokens_with = tokens
        start_idx, end_idx = len(tokens_without), len(tokens_with)
        if start_idx >= end_idx or tokens_without != tokens_with[:start_idx]:
            raise ValueError("Language-modeling CORE prompt without continuation is not a prefix.")
        return [tokens_with], [start_idx], [end_idx]
    raise ValueError(f"Unsupported task type: {task_type}")


def _fewshot_examples(data: list[dict[str, Any]], idx: int, num_fewshot: int) -> list[dict[str, Any]]:
    if num_fewshot <= 0:
        return []
    rng = random.Random(1234 + idx)
    available = [i for i in range(len(data)) if i != idx]
    return [data[i] for i in rng.sample(available, int(num_fewshot))]


def _stack_token_lists(token_lists: list[list[int]], *, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = max(len(tokens) for tokens in token_lists)
    out = torch.full((len(token_lists), seq_len), int(pad_token_id), dtype=torch.long)
    mask = torch.zeros((len(token_lists), seq_len), dtype=torch.bool)
    for i, tokens in enumerate(token_lists):
        out[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        mask[i, : len(tokens)] = True
    return out, mask


def _crop_token_lists(
    token_lists: list[list[int]],
    start_indices: list[int],
    end_indices: list[int],
    max_seq_len: int | None,
) -> tuple[list[list[int]], list[int], list[int], int]:
    if max_seq_len is None:
        return token_lists, start_indices, end_indices, 0
    cropped = 0
    new_tokens = []
    new_starts = []
    new_ends = []
    for tokens, start, end in zip(token_lists, start_indices, end_indices):
        if len(tokens) > max_seq_len:
            crop = len(tokens) - int(max_seq_len)
            tokens = tokens[-int(max_seq_len) :]
            start -= crop
            end -= crop
            cropped += 1
            if start < 0 or end < 0:
                raise ValueError("CORE crop removed the scored continuation; max_seq_len is too small.")
        new_tokens.append(tokens)
        new_starts.append(start)
        new_ends.append(end)
    return new_tokens, new_starts, new_ends, cropped


def _teacher_max_seq_len(teacher: TeacherRunner) -> int | None:
    if not hasattr(teacher, "_load_model"):
        return None
    model = teacher._load_model()
    candidates = []
    freqs_cis = getattr(model, "freqs_cis", None)
    if freqs_cis is not None and getattr(freqs_cis, "ndim", 0) >= 2:
        candidates.append(int(freqs_cis.shape[1]))
    config = getattr(model, "config", None)
    for name in ("block_size", "max_seq_len", "max_position_embeddings", "n_positions"):
        value = getattr(config, name, None) if config is not None else None
        if value is not None:
            candidates.append(int(value))
    candidates = [value for value in candidates if value > 0]
    return min(candidates) if candidates else None


def _crop_to_context(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    prefix_lengths: torch.Tensor | None,
    max_seq_len: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    if max_seq_len is None or tokens.shape[1] <= max_seq_len:
        return tokens, mask, prefix_lengths, 0
    crop = tokens.shape[1] - int(max_seq_len)
    tokens = tokens[:, crop:]
    mask = mask[:, crop:]
    if prefix_lengths is not None:
        prefix_lengths = prefix_lengths.clone()
        non_negative = prefix_lengths >= 0
        prefix_lengths[non_negative] = (prefix_lengths[non_negative] - crop).clamp_min(0)
    return tokens, mask, prefix_lengths, crop


def _valid_score_mask(tokens: torch.Tensor, mask: torch.Tensor, prefix_lengths: torch.Tensor | None, vocab: int) -> torch.Tensor:
    if tokens.shape[1] < 2:
        return torch.zeros_like(tokens[:, 1:], dtype=torch.bool)
    targets = tokens[:, 1:]
    valid = mask[:, 1:].bool() & (targets < vocab)
    if prefix_lengths is not None:
        pos = torch.arange(1, tokens.shape[1], device=tokens.device).unsqueeze(0)
        prefixes = prefix_lengths.to(device=tokens.device).reshape(-1, 1)
        valid = valid & ((prefixes < 0) | (pos >= prefixes))
    return valid


def _span_score_mask(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
    vocab: int,
) -> torch.Tensor:
    if tokens.shape[1] < 2:
        return torch.zeros_like(tokens[:, 1:], dtype=torch.bool)
    targets = tokens[:, 1:]
    pos = torch.arange(1, tokens.shape[1], device=tokens.device).unsqueeze(0)
    start = start_indices.to(device=tokens.device).reshape(-1, 1)
    end = end_indices.to(device=tokens.device).reshape(-1, 1)
    return mask[:, 1:].bool() & (targets < vocab) & (pos >= start) & (pos < end)


def sequence_nll(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    prefix_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.shape[1] < 2:
        zeros = logits.new_zeros((logits.shape[0],))
        return zeros, logits.new_zeros(()), logits.new_zeros(())
    vocab = logits.shape[-1]
    targets = tokens[:, 1:].to(device=logits.device, dtype=torch.long)
    valid = _valid_score_mask(tokens.to(logits.device), mask.to(logits.device), prefix_lengths, vocab)
    safe_targets = targets.clamp(min=0, max=max(vocab - 1, 0))
    per_token = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, vocab),
        safe_targets.reshape(-1),
        reduction="none",
    ).reshape_as(safe_targets)
    counts = valid.sum(dim=1).clamp_min(1)
    per_sequence = (per_token * valid.float()).sum(dim=1) / counts
    return per_sequence, (per_token * valid.float()).sum(), valid.sum()


def sequence_span_nll(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.shape[1] < 2:
        zeros = logits.new_zeros((logits.shape[0],))
        return zeros, logits.new_zeros(()), logits.new_zeros(())
    vocab = logits.shape[-1]
    targets = tokens[:, 1:].to(device=logits.device, dtype=torch.long)
    valid = _span_score_mask(
        tokens.to(logits.device),
        mask.to(logits.device),
        start_indices.to(logits.device),
        end_indices.to(logits.device),
        vocab,
    )
    safe_targets = targets.clamp(min=0, max=max(vocab - 1, 0))
    per_token = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, vocab),
        safe_targets.reshape(-1),
        reduction="none",
    ).reshape_as(safe_targets)
    counts = valid.sum(dim=1).clamp_min(1)
    per_sequence = (per_token * valid.float()).sum(dim=1) / counts
    return per_sequence, (per_token * valid.float()).sum(), valid.sum()


def _span_exact_match(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[1] < 2:
        return logits.new_tensor(False, dtype=torch.bool)
    vocab = logits.shape[-1]
    targets = tokens[:, 1:].to(device=logits.device, dtype=torch.long)
    valid = _span_score_mask(
        tokens.to(logits.device),
        mask.to(logits.device),
        start_indices.to(logits.device),
        end_indices.to(logits.device),
        vocab,
    )
    predicted = logits[:, :-1].argmax(dim=-1)
    correct = (predicted == targets.clamp(max=max(vocab - 1, 0))) | ~valid
    return correct.all(dim=1)


def _token_accuracy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    prefix_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[1] < 2:
        return logits.new_zeros(()), logits.new_zeros(())
    vocab = logits.shape[-1]
    targets = tokens[:, 1:].to(device=logits.device, dtype=torch.long)
    valid = _valid_score_mask(tokens.to(logits.device), mask.to(logits.device), prefix_lengths, vocab)
    correct = (logits[:, :-1].argmax(dim=-1) == targets.clamp(max=max(vocab - 1, 0))) & valid
    return correct.float().sum(), valid.sum()


def _kl_and_agreement(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    student_logits = student_logits[..., :vocab].float()
    teacher_logits = teacher_logits[..., :vocab].float()
    mask = mask.to(device=student_logits.device, dtype=torch.bool)
    t = float(temperature)
    log_p = F.log_softmax(student_logits / t, dim=-1)
    q = F.softmax(teacher_logits / t, dim=-1)
    kl = F.kl_div(log_p, q, reduction="none").sum(dim=-1) * (t * t)
    top1 = (student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)).float()
    k = min(int(top_k), vocab)
    if k > 0:
        student_top = student_logits.topk(k, dim=-1).indices
        teacher_top = teacher_logits.topk(k, dim=-1).indices
        topk = (student_top.unsqueeze(-1) == teacher_top.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1) / k
    else:
        topk = torch.zeros_like(top1)
    count = mask.sum()
    return {
        "kl": ((kl * mask.float()).sum(), count),
        "top1": ((top1 * mask.float()).sum(), count),
        "topk": ((topk * mask.float()).sum(), count),
    }


def _call_solve_forward(model: Any, context: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor, steps: int | None):
    if steps is None:
        try:
            return model._solve_forward(context, freqs_cis, mask)
        except TypeError:
            return model._solve_forward(context, freqs_cis, mask, max_iter_override=None)
    return model._solve_forward(context, freqs_cis, mask, max_iter_override=int(steps))


def teacher_logits_for_tokens(
    teacher: TeacherRunner,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    steps: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(teacher, "_load_model") or not hasattr(teacher, "project_logits"):
        raise TypeError("CORE eval currently requires an Attractor-like teacher with _load_model and project_logits.")
    model = teacher._load_model()
    device = teacher.device if hasattr(teacher, "device") else next(model.parameters()).device
    tokens = tokens.to(device=device, dtype=torch.long)
    attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    with torch.no_grad():
        freqs_cis = model.freqs_cis[:, : tokens.shape[1]]
        context = model._encode(tokens, freqs_cis, attention_mask)
        if steps == 0:
            state = context
            info = {"iters": 0.0, "rel_residual": 0.0}
        else:
            state, info = _call_solve_forward(model, context, freqs_cis, attention_mask, steps)
        logits = teacher.project_logits(state)
    solver_iters = torch.full((tokens.shape[0],), float(info.get("iters", -1.0)), device=device)
    rel_residual = torch.full((tokens.shape[0],), float(info.get("rel_residual", 0.0)), device=device)
    return context.detach(), state.detach(), logits.detach(), torch.stack((solver_iters, rel_residual), dim=1)


def rollout_student(
    student: nn.Module,
    z0: torch.Tensor,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    steps: int,
    mode: str,
) -> torch.Tensor:
    if steps <= 0:
        return z0
    current = z0
    tokens = tokens.to(device=current.device, dtype=torch.long)
    attention_mask = attention_mask.to(device=current.device, dtype=torch.bool)
    for step in range(steps):
        t = torch.full((current.shape[0],), step / steps, device=current.device, dtype=current.dtype)
        delta = torch.full((current.shape[0],), 1.0 / steps, device=current.device, dtype=current.dtype)
        call_mode = "flow_map" if mode in {"flow_map", "avg_velocity"} else "velocity"
        out = student(current, t, delta, tokens=tokens, attention_mask=attention_mask, mode=call_mode)
        if mode == "flow_map":
            if out.z_next is None:
                raise ValueError("rollout_mode='flow_map' requires student output z_next.")
            current = out.z_next
        elif mode == "avg_velocity":
            if out.avg_velocity is None:
                raise ValueError("rollout_mode='avg_velocity' requires student output avg_velocity.")
            current = current + delta.reshape(-1, 1, 1) * out.avg_velocity
        elif mode == "velocity":
            current = current + delta.reshape(-1, 1, 1) * out.velocity
        else:
            raise ValueError("rollout_mode must be one of: velocity, flow_map, avg_velocity.")
    return current


def _model_logits(
    teacher: TeacherRunner,
    student: nn.Module | None,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    cfg: DictConfig,
    *,
    teacher_steps: int | None,
    student_steps: int | None,
) -> LogitBatch:
    z0, _teacher_state, teacher_logits, info = teacher_logits_for_tokens(teacher, tokens, mask, steps=teacher_steps)
    student_logits = None
    if student is not None:
        if student_steps is None:
            raise ValueError("Student evaluation requires eval.student_steps=<int> or an inferable auto value.")
        student_dtype = _resolve_dtype(str(cfg.eval.student_dtype))
        if student_dtype is not None:
            z0 = z0.to(dtype=student_dtype)
        z_student = rollout_student(
            student,
            z0,
            tokens,
            mask,
            steps=int(student_steps),
            mode=str(cfg.eval.rollout_mode),
        )
        with torch.no_grad():
            student_logits = teacher.project_logits(z_student).detach()
    return LogitBatch(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        solver_iters=info[:, 0],
        rel_residual=info[:, 1],
    )


def _update_sequence_metrics(
    acc: MetricAccumulators,
    logits: LogitBatch,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    prefix_lengths: torch.Tensor | None,
    cfg: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    device_tokens = tokens.to(device=logits.teacher_logits.device, dtype=torch.long)
    device_mask = mask.to(device=logits.teacher_logits.device, dtype=torch.bool)
    device_prefix = None if prefix_lengths is None else prefix_lengths.to(logits.teacher_logits.device)
    teacher_seq_nll, teacher_sum, teacher_count = sequence_nll(
        logits.teacher_logits, device_tokens, device_mask, device_prefix
    )
    acc.teacher_nll.add(teacher_sum, teacher_count)
    teacher_correct, teacher_acc_count = _token_accuracy(logits.teacher_logits, device_tokens, device_mask, device_prefix)
    acc.teacher_next_token_accuracy.add(teacher_correct, teacher_acc_count)
    if logits.solver_iters is not None:
        acc.solver_iters.add(logits.solver_iters.sum(), logits.solver_iters.numel())
    if logits.rel_residual is not None:
        acc.rel_residual.add(logits.rel_residual.sum(), logits.rel_residual.numel())
    student_seq_nll = None
    if logits.student_logits is not None:
        student_seq_nll, student_sum, student_count = sequence_nll(
            logits.student_logits, device_tokens, device_mask, device_prefix
        )
        acc.student_nll.add(student_sum, student_count)
        student_correct, student_acc_count = _token_accuracy(
            logits.student_logits, device_tokens, device_mask, device_prefix
        )
        acc.student_next_token_accuracy.add(student_correct, student_acc_count)
        cmp_metrics = _kl_and_agreement(
            logits.student_logits,
            logits.teacher_logits,
            device_mask,
            temperature=float(cfg.eval.temperature),
            top_k=int(cfg.eval.top_k),
        )
        acc.kl.add(*cmp_metrics["kl"])
        acc.top1_agreement.add(*cmp_metrics["top1"])
        acc.topk_overlap.add(*cmp_metrics["topk"])
    return teacher_seq_nll.detach().cpu(), None if student_seq_nll is None else student_seq_nll.detach().cpu()


def _update_span_metrics(
    acc: MetricAccumulators,
    logits: LogitBatch,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
    cfg: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    device_tokens = tokens.to(device=logits.teacher_logits.device, dtype=torch.long)
    device_mask = mask.to(device=logits.teacher_logits.device, dtype=torch.bool)
    device_starts = start_indices.to(device=logits.teacher_logits.device)
    device_ends = end_indices.to(device=logits.teacher_logits.device)
    teacher_seq_nll, teacher_sum, teacher_count = sequence_span_nll(
        logits.teacher_logits, device_tokens, device_mask, device_starts, device_ends
    )
    acc.teacher_nll.add(teacher_sum, teacher_count)
    teacher_exact = _span_exact_match(logits.teacher_logits, device_tokens, device_mask, device_starts, device_ends)
    acc.teacher_next_token_accuracy.add(teacher_exact.float().sum(), teacher_exact.numel())
    if logits.solver_iters is not None:
        acc.solver_iters.add(logits.solver_iters.sum(), logits.solver_iters.numel())
    if logits.rel_residual is not None:
        acc.rel_residual.add(logits.rel_residual.sum(), logits.rel_residual.numel())
    student_seq_nll = None
    if logits.student_logits is not None:
        student_seq_nll, student_sum, student_count = sequence_span_nll(
            logits.student_logits, device_tokens, device_mask, device_starts, device_ends
        )
        acc.student_nll.add(student_sum, student_count)
        student_exact = _span_exact_match(logits.student_logits, device_tokens, device_mask, device_starts, device_ends)
        acc.student_next_token_accuracy.add(student_exact.float().sum(), student_exact.numel())
        cmp_metrics = _kl_and_agreement(
            logits.student_logits,
            logits.teacher_logits,
            device_mask,
            temperature=float(cfg.eval.temperature),
            top_k=int(cfg.eval.top_k),
        )
        acc.kl.add(*cmp_metrics["kl"])
        acc.top1_agreement.add(*cmp_metrics["top1"])
        acc.topk_overlap.add(*cmp_metrics["topk"])
    return teacher_seq_nll.detach().cpu(), None if student_seq_nll is None else student_seq_nll.detach().cpu()


def _limited_indices(total: int, limit: int | None) -> range:
    if limit is None:
        return range(total)
    return range(min(total, int(limit)))


def evaluate_lm_task(
    task: dict[str, Any],
    teacher: TeacherRunner,
    student: nn.Module | None,
    cfg: DictConfig,
    *,
    teacher_steps: int | None,
    student_steps: int | None,
    max_seq_len: int | None,
) -> dict[str, float | str | int]:
    tokens = task["tokens"].long()
    mask = task["attention_mask"].bool()
    prefix = task.get("prefix_lengths")
    acc = MetricAccumulators()
    n_items = len(_limited_indices(tokens.shape[0], cfg.eval.max_examples_per_task))
    cropped_sequences = 0
    for start in range(0, n_items, int(cfg.eval.sequence_batch_size)):
        end = min(start + int(cfg.eval.sequence_batch_size), n_items)
        prefix_batch = None if prefix is None else prefix[start:end]
        seq_tokens, seq_mask, seq_prefix, crop = _crop_to_context(
            tokens[start:end], mask[start:end], prefix_batch, max_seq_len
        )
        if crop:
            cropped_sequences += int(seq_tokens.shape[0])
        logits = _model_logits(
            teacher, student, seq_tokens, seq_mask, cfg, teacher_steps=teacher_steps, student_steps=student_steps
        )
        _update_sequence_metrics(acc, logits, seq_tokens, seq_mask, seq_prefix, cfg)
    metrics = acc.to_metrics(max_ppl_exp=float(cfg.eval.max_ppl_exp))
    metrics.update({"rows": n_items, "sequences": n_items, "cropped_sequences": cropped_sequences})
    return metrics


def _render_raw_example(
    item: dict[str, Any],
    task_type: str,
    continuation_delimiter: str,
    fewshot_examples: list[dict[str, Any]],
) -> list[str]:
    if task_type == "multiple_choice":
        return _render_prompts_mc(item, continuation_delimiter, fewshot_examples)
    if task_type == "schema":
        return _render_prompts_schema(item, continuation_delimiter, fewshot_examples)
    if task_type == "language_modeling":
        return _render_prompts_lm(item, continuation_delimiter, fewshot_examples)
    raise ValueError(f"Unsupported task type: {task_type}")


def evaluate_raw_core_task(
    task_meta: dict[str, Any],
    data: list[dict[str, Any]],
    tokenizer: Any,
    teacher: TeacherRunner,
    student: nn.Module | None,
    cfg: DictConfig,
    *,
    teacher_steps: int | None,
    student_steps: int | None,
    max_seq_len: int | None,
) -> dict[str, float | str | int]:
    task_type = str(task_meta["icl_task_type"])
    continuation_delimiter = str(task_meta.get("continuation_delimiter", ""))
    num_fewshot_value = task_meta.get("num_fewshot", [0])
    if isinstance(num_fewshot_value, (list, tuple)):
        num_fewshot = int(num_fewshot_value[0])
    else:
        num_fewshot = int(num_fewshot_value)
    if not bool(cfg.eval.use_fewshot):
        num_fewshot = 0
    n_items = len(_limited_indices(len(data), cfg.eval.max_examples_per_task))
    acc = MetricAccumulators()
    correct_teacher = 0
    correct_student = 0
    cropped_sequences = 0
    total_sequences = 0
    pad_token_id = _pad_token_id(tokenizer, cfg)

    for idx in range(n_items):
        item = data[idx]
        fewshot = _fewshot_examples(data, idx, num_fewshot)
        prompts = _render_raw_example(item, task_type, continuation_delimiter, fewshot)
        token_lists, start_indices, end_indices = _batch_sequences(tokenizer, prompts, task_type, cfg)
        token_lists, start_indices, end_indices, cropped = _crop_token_lists(
            token_lists, start_indices, end_indices, max_seq_len
        )
        cropped_sequences += cropped
        tokens, mask = _stack_token_lists(token_lists, pad_token_id=pad_token_id)
        starts = torch.tensor(start_indices, dtype=torch.int32)
        ends = torch.tensor(end_indices, dtype=torch.int32)
        logits = _model_logits(teacher, student, tokens, mask, cfg, teacher_steps=teacher_steps, student_steps=student_steps)
        teacher_nll, student_nll = _update_span_metrics(acc, logits, tokens, mask, starts, ends, cfg)
        total_sequences += int(tokens.shape[0])

        if task_type == "language_modeling":
            teacher_exact = _span_exact_match(
                logits.teacher_logits,
                tokens.to(logits.teacher_logits.device),
                mask.to(logits.teacher_logits.device),
                starts.to(logits.teacher_logits.device),
                ends.to(logits.teacher_logits.device),
            )
            correct_teacher += int(teacher_exact[0].item())
            if logits.student_logits is not None:
                student_exact = _span_exact_match(
                    logits.student_logits,
                    tokens.to(logits.student_logits.device),
                    mask.to(logits.student_logits.device),
                    starts.to(logits.student_logits.device),
                    ends.to(logits.student_logits.device),
                )
                correct_student += int(student_exact[0].item())
        else:
            gold = int(item["gold"])
            teacher_pred = int(torch.argmin(teacher_nll).item())
            correct_teacher += int(teacher_pred == gold)
            if student_nll is not None:
                student_pred = int(torch.argmin(student_nll).item())
                correct_student += int(student_pred == gold)

    metrics = acc.to_metrics(max_ppl_exp=float(cfg.eval.max_ppl_exp))
    teacher_accuracy = correct_teacher / max(n_items, 1)
    metrics.update(
        {
            "rows": n_items,
            "sequences": total_sequences,
            "cropped_sequences": cropped_sequences,
            "teacher_accuracy": teacher_accuracy,
            "fewshot": num_fewshot,
        }
    )
    if student is not None:
        student_accuracy = correct_student / max(n_items, 1)
        metrics["student_accuracy"] = student_accuracy
        metrics["accuracy_delta"] = student_accuracy - teacher_accuracy
    return metrics


def evaluate_candidate_task(
    task: dict[str, Any],
    teacher: TeacherRunner,
    student: nn.Module | None,
    cfg: DictConfig,
    *,
    teacher_steps: int | None,
    student_steps: int | None,
    max_seq_len: int | None,
) -> dict[str, float | str | int]:
    tokens = task["candidate_tokens"].long()
    mask = task["candidate_attention_mask"].bool()
    choice_mask = task["choice_mask"].bool()
    gold = task["gold"].long()
    prefix = task.get("prefix_lengths")
    n_rows = len(_limited_indices(tokens.shape[0], cfg.eval.max_examples_per_task))
    n_choices = tokens.shape[1]
    teacher_scores = torch.full((n_rows, n_choices), float("inf"))
    student_scores = torch.full((n_rows, n_choices), float("inf")) if student is not None else None
    acc = MetricAccumulators()
    cropped_sequences = 0

    valid_pairs = [(i, j) for i in range(n_rows) for j in range(n_choices) if bool(choice_mask[i, j])]
    for start in range(0, len(valid_pairs), int(cfg.eval.sequence_batch_size)):
        pairs = valid_pairs[start : start + int(cfg.eval.sequence_batch_size)]
        row_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        choice_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        seq_tokens = tokens[row_idx, choice_idx]
        seq_mask = mask[row_idx, choice_idx]
        seq_prefix = None if prefix is None else prefix[row_idx]
        seq_tokens, seq_mask, seq_prefix, crop = _crop_to_context(seq_tokens, seq_mask, seq_prefix, max_seq_len)
        if crop:
            cropped_sequences += int(seq_tokens.shape[0])
        logits = _model_logits(
            teacher, student, seq_tokens, seq_mask, cfg, teacher_steps=teacher_steps, student_steps=student_steps
        )
        teacher_nll, student_nll = _update_sequence_metrics(acc, logits, seq_tokens, seq_mask, seq_prefix, cfg)
        for local, (row, choice) in enumerate(pairs):
            teacher_scores[row, choice] = teacher_nll[local]
            if student_scores is not None and student_nll is not None:
                student_scores[row, choice] = student_nll[local]

    valid_gold = (gold[:n_rows] >= 0) & (gold[:n_rows] < n_choices)
    teacher_pred = teacher_scores.argmin(dim=1)
    teacher_accuracy = ((teacher_pred == gold[:n_rows]) & valid_gold).float().sum() / valid_gold.float().sum().clamp_min(1)
    metrics = acc.to_metrics(max_ppl_exp=float(cfg.eval.max_ppl_exp))
    metrics.update(
        {
            "rows": n_rows,
            "sequences": int(choice_mask[:n_rows].sum().item()),
            "cropped_sequences": cropped_sequences,
            "teacher_accuracy": float(teacher_accuracy.item()),
        }
    )
    if student_scores is not None:
        student_pred = student_scores.argmin(dim=1)
        student_accuracy = ((student_pred == gold[:n_rows]) & valid_gold).float().sum() / valid_gold.float().sum().clamp_min(1)
        metrics["student_accuracy"] = float(student_accuracy.item())
        metrics["accuracy_delta"] = float(student_accuracy.item() - teacher_accuracy.item())
    return metrics


def _task_files(tokenized_dir: Path, cfg: DictConfig) -> list[Path]:
    metadata_path = tokenized_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        files = [tokenized_dir / task["output_file"] for task in metadata.get("tasks", [])]
    else:
        files = sorted(path for path in tokenized_dir.glob("*.pt") if path.name != "metadata.pt")
    selected = cfg.eval.tasks
    if selected:
        if isinstance(selected, str):
            selected = [selected]
        wanted = {str(item) for item in selected}
        files = [path for path in files if path.stem in wanted]
    if cfg.eval.max_tasks is not None:
        files = files[: int(cfg.eval.max_tasks)]
    return files


def _raw_core_tasks(eval_bundle_dir: Path, cfg: DictConfig) -> list[dict[str, Any]]:
    core_cfg = OmegaConf.to_container(OmegaConf.load(eval_bundle_dir / "core.yaml"), resolve=True)
    tasks = list(core_cfg["icl_tasks"])
    selected = cfg.eval.tasks
    if selected:
        if isinstance(selected, str):
            selected = [selected]
        wanted = {str(item) for item in selected}
        tasks = [task for task in tasks if str(task["label"]) in wanted]
    if cfg.eval.max_tasks is not None:
        tasks = tasks[: int(cfg.eval.max_tasks)]
    return tasks


def _write_task_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    aggregate: dict[str, float] = {"tasks": float(len(rows))}
    numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    for key in numeric_keys:
        if key in {"rows", "sequences"}:
            aggregate[key] = float(sum(float(row.get(key, 0.0)) for row in rows))
        else:
            values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
            if values:
                aggregate[f"macro_{key}"] = sum(values) / len(values)
    return aggregate


def run(cfg: DictConfig) -> None:
    started = time()
    dirs = ensure_run_dirs(cfg.output_dir)
    save_resolved_config(cfg, cfg.output_dir)
    torch.manual_seed(int(cfg.seed))

    tokenized_dir = Path(str(cfg.eval.tokenized_dir)).expanduser()
    if not tokenized_dir.exists() and not cfg.eval.eval_bundle_dir:
        raise FileNotFoundError(f"Tokenized CORE directory not found: {tokenized_dir}")
    device = _resolve_device(str(cfg.eval.device))
    cfg.teacher.device = str(device)
    teacher = instantiate(cfg.teacher)
    if hasattr(teacher, "set_device"):
        teacher.set_device(device)
    max_seq_len = _teacher_max_seq_len(teacher)
    if max_seq_len is not None:
        info(f"Teacher max sequence length: {max_seq_len}")

    model_kind = str(cfg.eval.model)
    if model_kind not in {"teacher", "student", "both"}:
        raise ValueError("eval.model must be one of: teacher, student, both.")
    student = None
    student_steps = None
    if model_kind in {"student", "both"}:
        if not cfg.eval.student_checkpoint:
            raise ValueError("Student evaluation requires eval.student_checkpoint=/path/to/checkpoint.ckpt")
        student = instantiate(_student_cfg(cfg))
        _load_student_checkpoint(student, str(cfg.eval.student_checkpoint), strict=bool(cfg.eval.strict_student_load))
        student_dtype = _resolve_dtype(str(cfg.eval.student_dtype))
        if student_dtype is not None:
            student = student.to(dtype=student_dtype)
        student = student.to(device=device)
        student.eval()
        student_steps = _as_step(cfg.eval.student_steps, auto_value=_student_auto_steps(cfg))
        if student_steps is None:
            warn("Could not infer student_steps from checkpoint config; falling back to teacher.max_depth.")
            student_steps = int(getattr(teacher, "max_depth", 1))

    teacher_steps = _as_step(cfg.eval.teacher_steps, auto_value=None)
    if teacher_steps is None:
        info("Teacher inference steps: auto solver")
    else:
        info(f"Teacher inference steps: forced {teacher_steps}")
    if student is not None:
        info(f"Student inference steps: {student_steps} ({cfg.eval.rollout_mode})")

    rows: list[dict[str, Any]] = []
    eval_bundle_dir = _resolve_eval_bundle_dir(tokenized_dir, cfg)
    if eval_bundle_dir is None and not tokenized_dir.exists():
        raise FileNotFoundError(
            "No CORE eval bundle or tokenized fallback found. Set eval.eval_bundle_dir or eval.tokenized_dir."
        )
    if bool(cfg.eval.prefer_raw_eval_bundle) and eval_bundle_dir is not None:
        tokenizer = _load_core_tokenizer(teacher)
        info(f"Using CORE raw eval_bundle: {eval_bundle_dir}")
        info(f"CORE prepend_bos={bool(cfg.eval.prepend_bos)} pad_token_id={_pad_token_id(tokenizer, cfg)}")
        for task_meta in _raw_core_tasks(eval_bundle_dir, cfg):
            label = str(task_meta["label"])
            info(f"Evaluating CORE task {label}")
            data = _read_jsonl(eval_bundle_dir / "eval_data" / str(task_meta["dataset_uri"]))
            metrics = evaluate_raw_core_task(
                task_meta,
                data,
                tokenizer,
                teacher,
                student,
                cfg,
                teacher_steps=teacher_steps,
                student_steps=student_steps,
                max_seq_len=max_seq_len,
            )
            rows.append(
                {
                    "task": label,
                    "task_type": str(task_meta.get("icl_task_type", "unknown")),
                    "seq_len": None,
                    **metrics,
                }
            )
    else:
        warn("Falling back to pretokenized CORE rows; this path does not reconstruct few-shot CORE prompts.")
        for path in _task_files(tokenized_dir, cfg):
            info(f"Evaluating CORE task {path.stem}")
            task = _load_pt(path)
            if "candidate_tokens" in task:
                metrics = evaluate_candidate_task(
                    task,
                    teacher,
                    student,
                    cfg,
                    teacher_steps=teacher_steps,
                    student_steps=student_steps,
                    max_seq_len=max_seq_len,
                )
            else:
                metrics = evaluate_lm_task(
                    task,
                    teacher,
                    student,
                    cfg,
                    teacher_steps=teacher_steps,
                    student_steps=student_steps,
                    max_seq_len=max_seq_len,
                )
            row = {
                "task": task.get("task_label", path.stem),
                "task_type": task.get("task_type", "unknown"),
                "seq_len": int(task.get("seq_len", 0)),
                **metrics,
            }
            rows.append(row)

    _write_task_metrics(dirs["metrics"] / "core_tasks.csv", rows)
    summary = _aggregate(rows)
    summary.update(
        {
            "teacher_steps": "auto" if teacher_steps is None else int(teacher_steps),
            "student_steps": None if student_steps is None else int(student_steps),
            "rollout_mode": str(cfg.eval.rollout_mode),
            "prepend_bos": bool(cfg.eval.prepend_bos),
            "pad_token_id": _pad_token_id(tokenizer, cfg),
            "eval_protocol": "core_comparable" if bool(cfg.eval.prepend_bos) else "attractor_distill",
        }
    )
    (dirs["metrics"] / "core_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_run_summary(cfg.output_dir, title="LoopDistill CORE evaluation", metrics=summary, started_at=started)
    ok(f"CORE evaluation complete: {cfg.output_dir}")


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            """LoopDistill CORE evaluation

Usage:
  uv run loopdistill-eval-core [Hydra overrides]

Common overrides:
  eval.tokenized_dir=/workspace/loopdistill_tokens/<core_tokenized_dir>
  eval.model=teacher
  eval.teacher_steps=auto
  eval.teacher_steps=16

  eval.model=student
  eval.student_checkpoint=/workspace/looped-transformer-distillation/outputs/.../checkpoints/<file>.ckpt
  eval.student_steps=auto
  eval.student_steps=4
  eval.rollout_mode=flow_map

  eval.model=both
  eval.tasks=[arc_easy,lambada_openai]
  eval.eval_bundle_dir=/workspace/datasets/<core_download_dir>/eval_bundle
  eval.prepend_bos=true
  eval.max_examples_per_task=100
  eval.sequence_batch_size=4
  output_dir=outputs/core_eval/<run_id>

Outputs:
  metrics/core_tasks.csv
  metrics/core_summary.json
  reports/run_summary.md
"""
        )
        return
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="eval_core", overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
