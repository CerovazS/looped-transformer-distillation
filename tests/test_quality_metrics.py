from __future__ import annotations

import torch
from torch import nn

from loopdistill.metrics.quality import QualityEvaluator
from loopdistill.models.student import StudentOutput


class FixedLogitStudent(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, z_t, t, delta, **kwargs):
        return StudentOutput(
            velocity=torch.zeros_like(z_t),
            z_next=z_t,
            avg_velocity=torch.zeros_like(z_t),
            logits=self.fixed_logits.to(device=z_t.device),
        )


class FixedProjectionTeacher:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def project_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.logits.to(device=z.device)


class ModeStepStudent(nn.Module):
    def forward(self, z_t, t, delta, **kwargs):
        ones = torch.ones_like(z_t)
        return StudentOutput(
            velocity=ones,
            z_next=z_t + 2.0 * delta.reshape(-1, 1, 1) * ones,
            avg_velocity=3.0 * ones,
            logits=torch.zeros(z_t.shape[0], z_t.shape[1], 4, device=z_t.device),
        )


def test_quality_metrics_identical_logits():
    logits = torch.tensor(
        [
            [
                [8.0, 1.0, 0.0, -1.0],
                [0.0, 7.0, 1.0, -2.0],
                [-1.0, 0.0, 6.0, 1.0],
            ]
        ]
    )
    batch = {
        "tokens": torch.tensor([[1, 1, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "z": torch.zeros(1, 2, 3, 5),
        "logits": logits.unsqueeze(1),
    }
    evaluator = QualityEvaluator(enabled=True, top_k=2)
    metrics = evaluator.compute(batch, FixedLogitStudent(logits))

    assert metrics["kl_student_teacher"] < 1e-6
    assert torch.isclose(metrics["nll_delta"], torch.zeros(()), atol=1e-6)
    assert torch.isclose(metrics["ppl_delta"], torch.zeros(()), atol=1e-5)
    assert torch.isclose(metrics["top1_agreement"], torch.ones(()))
    assert torch.isclose(metrics["top2_overlap"], torch.ones(()))


def test_quality_metrics_student_worse_than_teacher_nll():
    teacher_logits = torch.tensor(
        [
            [
                [0.0, 9.0, 0.0],
                [0.0, 0.0, 9.0],
                [9.0, 0.0, 0.0],
            ]
        ]
    )
    student_logits = torch.zeros_like(teacher_logits)
    batch = {
        "tokens": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "z": torch.zeros(1, 2, 3, 4),
        "logits": teacher_logits.unsqueeze(1),
    }
    evaluator = QualityEvaluator(enabled=True, top_k=1)
    metrics = evaluator.compute(batch, FixedLogitStudent(student_logits))

    assert metrics["nll_student"] > metrics["nll_teacher"]
    assert metrics["ppl_student"] > metrics["ppl_teacher"]
    assert metrics["nll_delta"] > 0
    assert metrics["top1_agreement"] < 1


def test_quality_metrics_can_compare_student_and_teacher_heads():
    teacher_logits = torch.tensor(
        [
            [
                [0.0, 8.0, 0.0],
                [0.0, 0.0, 8.0],
                [8.0, 0.0, 0.0],
            ]
        ]
    )
    student_head_logits = torch.zeros_like(teacher_logits)
    batch = {
        "tokens": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "z": torch.zeros(1, 2, 3, 4),
        "logits": teacher_logits.unsqueeze(1),
    }
    evaluator = QualityEvaluator(enabled=True, top_k=1, projections=["student_head", "teacher_head"])
    metrics = evaluator.compute(
        batch,
        FixedLogitStudent(student_head_logits),
        teacher=FixedProjectionTeacher(teacher_logits),
    )

    assert "student_head/nll_student" in metrics
    assert "teacher_head/nll_student" in metrics
    assert metrics["student_head/nll_student"] > metrics["student_head/nll_teacher"]
    assert torch.isclose(metrics["teacher_head/nll_delta"], torch.zeros(()), atol=1e-6)
    assert torch.isclose(metrics["teacher_head/top1_agreement"], torch.ones(()))


def test_quality_rollout_modes_select_expected_student_head():
    batch = {
        "tokens": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "z": torch.zeros(1, 2, 3, 4),
        "logits": torch.zeros(1, 1, 3, 4),
    }
    student = ModeStepStudent()

    velocity_z = QualityEvaluator(enabled=True, rollout_steps=1, rollout_mode="velocity")._rollout(student, batch)
    flow_map_z = QualityEvaluator(enabled=True, rollout_steps=1, rollout_mode="flow_map")._rollout(student, batch)
    avg_velocity_z = QualityEvaluator(enabled=True, rollout_steps=1, rollout_mode="avg_velocity")._rollout(student, batch)

    assert torch.allclose(velocity_z, torch.ones_like(velocity_z))
    assert torch.allclose(flow_map_z, torch.full_like(flow_map_z, 2.0))
    assert torch.allclose(avg_velocity_z, torch.full_like(avg_velocity_z, 3.0))
