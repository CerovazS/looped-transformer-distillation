from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from loopdistill.losses.common import masked_kl, masked_mean


class QualityEvaluator(nn.Module):
    """Endpoint language-model quality metrics for teacher/student comparison."""

    def __init__(
        self,
        enabled: bool = False,
        top_k: int = 5,
        temperature: float = 1.0,
        rollout_steps: int | None = None,
        every_n_epochs: int = 1,
        run_on_test: bool = True,
        require_logits: bool = True,
        max_ppl_exp: float = 20.0,
        projections: str | list[str] | tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.rollout_steps = rollout_steps
        self.every_n_epochs = int(every_n_epochs)
        self.run_on_test = bool(run_on_test)
        self.require_logits = bool(require_logits)
        self.max_ppl_exp = float(max_ppl_exp)
        self.projections = self._normalize_projections(projections)

    def compute(
        self,
        batch: dict[str, Any],
        student: nn.Module,
        teacher: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return {}
        teacher_logits = self._teacher_endpoint_logits(batch, teacher=teacher)
        if teacher_logits is None:
            if self.require_logits:
                raise ValueError("eval_quality requires teacher endpoint logits, but batch['logits'] is missing.")
            return {}

        z_student = self._rollout(student, batch)
        metrics: dict[str, torch.Tensor] = {}
        for projection in self.projections:
            logits = self._project_student_logits(projection, student, teacher, z_student, batch)
            if logits is None:
                continue
            projection_metrics = self._compute_logits_metrics(logits, teacher_logits, batch)
            if self._use_legacy_names(projection):
                metrics.update(projection_metrics)
            else:
                metrics.update({f"{projection}/{key}": value for key, value in projection_metrics.items()})
        return metrics

    def _normalize_projections(self, projections: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
        if projections is None:
            return ("student_head",)
        if isinstance(projections, str):
            projections = (projections,)
        normalized = tuple(str(projection) for projection in projections)
        supported = {"student_head", "teacher_head"}
        unknown = sorted(set(normalized) - supported)
        if unknown:
            raise ValueError(f"Unsupported quality projections: {unknown}. Expected one of {sorted(supported)}.")
        if not normalized:
            raise ValueError("QualityEvaluator requires at least one projection.")
        return normalized

    def _use_legacy_names(self, projection: str) -> bool:
        return self.projections == ("student_head",) and projection == "student_head"

    def _project_student_logits(
        self,
        projection: str,
        student: nn.Module,
        teacher: Any | None,
        z_student: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor | None:
        z = batch["z"]
        tokens = batch["tokens"]
        mask = batch["attention_mask"]
        if projection == "student_head":
            dtype = z.dtype
            out = student(
                z_student,
                torch.ones(z.shape[0], device=z.device, dtype=dtype),
                torch.zeros(z.shape[0], device=z.device, dtype=dtype),
                tokens=tokens,
                attention_mask=mask,
            )
            if out.logits is None:
                if self.require_logits:
                    raise ValueError("eval_quality projection 'student_head' requires student.logit_head.")
                return None
            return out.logits
        if teacher is None or not hasattr(teacher, "project_logits"):
            if self.require_logits:
                raise ValueError("eval_quality projection 'teacher_head' requires a teacher.project_logits(z) method.")
            return None
        with torch.no_grad():
            return teacher.project_logits(z_student)
        raise AssertionError(f"Unhandled quality projection: {projection}")

    def _compute_logits_metrics(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        tokens = batch["tokens"]
        mask = batch["attention_mask"]
        student_logits, teacher_logits = self._align_vocab(student_logits.float(), teacher_logits.float())
        teacher_nll = self._next_token_nll(teacher_logits, tokens, mask)
        student_nll = self._next_token_nll(student_logits, tokens, mask)
        teacher_ppl = torch.exp(teacher_nll.clamp(max=self.max_ppl_exp))
        student_ppl = torch.exp(student_nll.clamp(max=self.max_ppl_exp))
        top1 = self._top1_agreement(student_logits, teacher_logits, mask)
        topk = self._topk_overlap(student_logits, teacher_logits, mask, self.top_k)

        return {
            "kl_student_teacher": masked_kl(student_logits, teacher_logits, mask, self.temperature).detach(),
            "nll_teacher": teacher_nll.detach(),
            "nll_student": student_nll.detach(),
            "nll_delta": (student_nll - teacher_nll).detach(),
            "ppl_teacher": teacher_ppl.detach(),
            "ppl_student": student_ppl.detach(),
            "ppl_delta": (student_ppl - teacher_ppl).detach(),
            "top1_agreement": top1.detach(),
            f"top{min(self.top_k, student_logits.shape[-1])}_overlap": topk.detach(),
        }

    def needs_batch_teacher_logits(self, teacher: Any | None = None) -> bool:
        return not (teacher is not None and hasattr(teacher, "project_logits"))

    def _teacher_endpoint_logits(self, batch: dict[str, Any], teacher: Any | None = None) -> torch.Tensor | None:
        logits = batch.get("logits")
        if logits is not None:
            if logits.dim() == 4:
                return logits[:, -1]
            if logits.dim() == 3:
                return logits
            raise ValueError(f"Expected teacher logits with shape [B,K,L,V] or [B,L,V], got {tuple(logits.shape)}.")
        if teacher is None or not hasattr(teacher, "project_logits"):
            return None
        z = batch.get("z")
        if z is None:
            return None
        with torch.no_grad():
            return teacher.project_logits(z[:, -1])

    def _rollout(self, student: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
        z = batch["z"]
        current = z[:, 0]
        tokens = batch["tokens"]
        mask = batch["attention_mask"]
        steps = int(self.rollout_steps or (z.shape[1] - 1))
        steps = max(steps, 1)
        for step in range(steps):
            t = torch.full((current.shape[0],), step / steps, device=current.device, dtype=current.dtype)
            delta = torch.full((current.shape[0],), 1.0 / steps, device=current.device, dtype=current.dtype)
            out = student(current, t, delta, tokens=tokens, attention_mask=mask, mode="velocity")
            current = current + delta.reshape(-1, 1, 1) * out.velocity
        return current

    def _next_token_nll(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if logits.shape[1] < 2:
            return logits.new_zeros(())
        vocab = logits.shape[-1]
        targets = tokens[:, 1:].to(device=logits.device, dtype=torch.long)
        valid = targets < vocab
        if mask is not None:
            valid = valid & mask[:, 1:].to(device=logits.device, dtype=torch.bool)
        safe_targets = targets.clamp(min=0, max=max(vocab - 1, 0))
        per_token = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab),
            safe_targets.reshape(-1),
            reduction="none",
        ).reshape_as(safe_targets)
        return masked_mean(per_token, valid)

    def _top1_agreement(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        agree = (student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)).float()
        return masked_mean(agree, mask)

    def _topk_overlap(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor | None,
        top_k: int,
    ) -> torch.Tensor:
        k = min(int(top_k), student_logits.shape[-1])
        if k <= 0:
            return student_logits.new_zeros(())
        student_top = student_logits.topk(k, dim=-1).indices
        teacher_top = teacher_logits.topk(k, dim=-1).indices
        overlap = (student_top.unsqueeze(-1) == teacher_top.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1) / k
        return masked_mean(overlap, mask)

    def _align_vocab(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
        if vocab <= 0:
            raise ValueError("Logits must have a non-empty vocabulary dimension.")
        return student_logits[..., :vocab], teacher_logits[..., :vocab]

    forward = compute
