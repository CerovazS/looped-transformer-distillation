from __future__ import annotations

import torch
import lightning as L
from hydra import compose, initialize_config_dir
from pathlib import Path
from torch.utils.data import DataLoader

from loopdistill.losses.flowmap import (
    CompositionalFlowMapLoss,
    EulerianMeanFlowTrajectoryLoss,
    LagrangianFlowMapLoss,
)
from loopdistill.losses.p0 import LoopDistillationLoss
from loopdistill.metrics.quality import QualityEvaluator
from loopdistill.models.student import StudentFlowModel
from loopdistill.teachers.base import TeacherOutput
from loopdistill.teachers.mock import MockTeacher
from loopdistill.train_module import DistillationModule


def test_live_teacher_step_from_token_batch_backward():
    student = StudentFlowModel(
        latent_dim=8,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        vocab_size=16,
    )
    teacher = MockTeacher(latent_dim=8, vocab_size=16, max_depth=2)
    module = DistillationModule(
        student=student,
        loss_module=LoopDistillationLoss(endpoint_kl_weight=0.0),
        teacher=teacher,
        live_depths=[0, 1, 2],
    )
    batch = {
        "tokens": torch.randint(0, 16, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
    }

    loss = module._step(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in student.parameters())


def test_live_teacher_step_with_flow_map_losses_backward():
    loss_modules = [
        LagrangianFlowMapLoss(),
        CompositionalFlowMapLoss(),
        EulerianMeanFlowTrajectoryLoss(),
    ]
    for loss_module in loss_modules:
        student = StudentFlowModel(
            latent_dim=8,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            vocab_size=16,
        )
        teacher = MockTeacher(latent_dim=8, vocab_size=16, max_depth=2)
        module = DistillationModule(
            student=student,
            loss_module=loss_module,
            teacher=teacher,
            live_depths=[0, 1, 2],
        )
        batch = {
            "tokens": torch.randint(0, 16, (2, 5)),
            "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        }

        loss = module._step(batch, "train")
        loss.backward()

        assert torch.isfinite(loss)
        assert any(param.grad is not None for param in student.parameters())


def test_live_optimizer_uses_student_parameters_only():
    student = StudentFlowModel(latent_dim=8, hidden_dim=32, num_layers=1, num_heads=4, vocab_size=16)
    module = DistillationModule(
        student=student,
        loss_module=LoopDistillationLoss(),
        teacher=MockTeacher(latent_dim=8, vocab_size=16),
    )

    optimizer = module.configure_optimizers()
    opt_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    student_param_ids = {id(param) for param in student.parameters()}

    assert opt_param_ids == student_param_ids


class _FlaggedLogitTeacher:
    teacher_id = "flagged"

    def __init__(self):
        self.return_logits = True
        self.calls: list[bool] = []

    def run_batch(self, tokens, attention_mask, depths):
        self.calls.append(bool(self.return_logits))
        batch, seq_len = tokens.shape
        depth = max(depths) + 1
        z = torch.randn(batch, depth, seq_len, 8, device=tokens.device)
        logits = torch.randn(batch, 1, seq_len, 16, device=tokens.device) if self.return_logits else None
        return TeacherOutput(
            z=z,
            logits=logits,
            loss_K=torch.zeros(batch, device=tokens.device),
            residual_norm=torch.zeros(batch, device=tokens.device),
            solver_iters=torch.full((batch,), depth - 1, device=tokens.device),
        )


class _ProjectingFlaggedLogitTeacher(_FlaggedLogitTeacher):
    def project_logits(self, z):
        batch, seq_len, _ = z.shape
        return torch.randn(batch, seq_len, 16, device=z.device)


def test_live_teacher_logits_only_requested_for_quality_eval():
    student = StudentFlowModel(latent_dim=8, hidden_dim=32, num_layers=1, num_heads=4, vocab_size=16)
    teacher = _FlaggedLogitTeacher()
    module = DistillationModule(
        student=student,
        loss_module=LoopDistillationLoss(endpoint_kl_weight=0.0),
        quality_evaluator=QualityEvaluator(enabled=True),
        teacher=teacher,
        live_depths=[0, 1],
    )
    batch = {
        "tokens": torch.randint(0, 16, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
    }

    module._step(dict(batch), "train")
    prepared = module._prepare_batch(dict(batch), "val")

    assert teacher.calls == [False, True]
    assert prepared["logits"] is not None


def test_live_teacher_logits_not_requested_when_teacher_can_project():
    student = StudentFlowModel(latent_dim=8, hidden_dim=32, num_layers=1, num_heads=4, vocab_size=16)
    teacher = _ProjectingFlaggedLogitTeacher()
    module = DistillationModule(
        student=student,
        loss_module=LoopDistillationLoss(endpoint_kl_weight=0.0),
        quality_evaluator=QualityEvaluator(enabled=True, projections=["student_head", "teacher_head"]),
        teacher=teacher,
        live_depths=[0, 1],
    )
    batch = {
        "tokens": torch.randint(0, 16, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
    }

    prepared = module._prepare_batch(dict(batch), "val")
    metrics = module.quality_evaluator.compute(prepared, module.student, teacher=teacher)

    assert teacher.calls == [False]
    assert prepared["logits"] is None
    assert "teacher_head/nll_student" in metrics


def test_live_quality_eval_can_use_teacher_head_projection():
    student = StudentFlowModel(latent_dim=8, hidden_dim=32, num_layers=1, num_heads=4, vocab_size=16)
    teacher = MockTeacher(latent_dim=8, vocab_size=16, max_depth=2)
    module = DistillationModule(
        student=student,
        loss_module=LoopDistillationLoss(endpoint_kl_weight=0.0),
        quality_evaluator=QualityEvaluator(enabled=True, projections=["student_head", "teacher_head"]),
        teacher=teacher,
        live_depths=[0, 1, 2],
    )
    batch = {
        "tokens": torch.randint(0, 16, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
    }

    prepared = module._prepare_batch(batch, "val")
    metrics = module.quality_evaluator.compute(prepared, module.student, teacher=teacher)

    assert "student_head/nll_student" in metrics
    assert "teacher_head/nll_student" in metrics


class _TokenOnlyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "tokens": torch.randint(0, 16, (5,)),
            "attention_mask": torch.ones(5, dtype=torch.bool),
        }


def _token_collate(batch):
    return {
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }


class _TokenOnlyDataModule(L.LightningDataModule):
    def setup(self, stage=None):
        self.dataset = _TokenOnlyDataset()

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=2, collate_fn=_token_collate)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=2, collate_fn=_token_collate)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=2, collate_fn=_token_collate)


def test_lightning_live_teacher_two_batches(tmp_path):
    student = StudentFlowModel(
        latent_dim=8,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        vocab_size=16,
    )
    module = DistillationModule(
        student=student,
        loss_module=LoopDistillationLoss(endpoint_kl_weight=0.0),
        teacher=MockTeacher(latent_dim=8, vocab_size=16, max_depth=2),
        live_depths=[0, 1, 2],
        metrics_dir=str(tmp_path / "metrics"),
    )
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=2,
        limit_val_batches=1,
        enable_model_summary=False,
    )

    trainer.fit(module, datamodule=_TokenOnlyDataModule())

    assert (tmp_path / "metrics" / "train.csv").exists()


def test_lightning_live_teacher_can_rename_final_test_metrics(tmp_path):
    student = StudentFlowModel(
        latent_dim=8,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        vocab_size=16,
    )
    module = DistillationModule(
        student=student,
        loss_module=LoopDistillationLoss(endpoint_kl_weight=0.0),
        quality_evaluator=QualityEvaluator(enabled=True, projections=["teacher_head"]),
        teacher=MockTeacher(latent_dim=8, vocab_size=16, max_depth=2),
        live_depths=[0, 1, 2],
        metrics_dir=str(tmp_path / "metrics"),
        test_metric_prefix="in_distribution_test",
    )
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(module, datamodule=_TokenOnlyDataModule())
    trainer.test(module, datamodule=_TokenOnlyDataModule())

    assert "in_distribution_test/loss" in trainer.callback_metrics
    assert "eval_quality/in_distribution_test/teacher_head/nll_delta" in trainer.callback_metrics
    assert (tmp_path / "metrics" / "in_distribution_test.csv").exists()


def test_hydra_live_experiment_overrides_defaults():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=blackwell_live_attractor140"])

    assert cfg.live.enabled is True
    assert cfg.data._target_ == "loopdistill.data.live.TextTokenDataModule"
    assert cfg.teacher._target_ == "loopdistill.teachers.attractor.AttractorTeacher"
    assert cfg.teacher.return_logits is False
    assert cfg.eval_quality.enabled is True
    assert cfg.eval_quality.projections == ["student_head", "teacher_head"]
    assert cfg.output_dir.startswith("outputs/live_distill/")


def test_hydra_full_live_experiment_uses_ddp_and_k8():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=blackwell_live_attractor140_p0_full"])

    assert cfg.trainer.devices == 2
    assert cfg.trainer.strategy == "ddp_find_unused_parameters_true"
    assert cfg.live.depths == list(range(9))
    assert cfg.loss.rollout_steps == 8
    assert cfg.loss.rollout_mode == "velocity"
    assert cfg.teacher.return_logits is False
    assert cfg.eval_quality.projections == ["student_head", "teacher_head"]
    assert cfg.eval_quality.rollout_mode == "velocity"
    assert cfg.data._target_ == "loopdistill.data.token_shards.TokenShardDataModule"
    assert len({cfg.tokenization.train_split, cfg.tokenization.val_split, cfg.tokenization.test_split}) == 3


def test_hydra_flow_map_loss_configs_compose():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    expected_targets = {
        "compositional": "loopdistill.losses.flowmap.CompositionalFlowMapLoss",
        "lagrangian": "loopdistill.losses.flowmap.LagrangianFlowMapLoss",
        "eulerian_meanflow": "loopdistill.losses.flowmap.EulerianMeanFlowTrajectoryLoss",
    }
    for loss_name, target in expected_targets.items():
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=[f"loss={loss_name}"])
        assert cfg.loss._target_ == target


def test_hydra_compositional_k16s8_experiment_compose():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=blackwell_live_attractor140_compositional_k16s8"])

    assert cfg.loss._target_ == "loopdistill.losses.flowmap.CompositionalFlowMapLoss"
    assert cfg.live.depths == list(range(17))
    assert cfg.teacher.max_depth == 16
    assert cfg.eval_quality.rollout_steps == 8
    assert cfg.eval_quality.rollout_mode == "flow_map"
    assert cfg.data.batch_size == 8


def test_hydra_huginn_recurrent_uses_in_distribution_final_test():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=blackwell_live_huginn_recurrent_k16s8"])

    assert cfg.data._target_ == "loopdistill.data.huginn_parquet.HuginnParquetDataModule"
    assert cfg.data.input_length is None
    assert cfg.eval_quality.projections == ["teacher_head"]
    assert cfg.eval_quality.every_n_epochs == 1
    assert cfg.data.train_samples is None
    assert cfg.trainer.val_check_interval == 200
    assert cfg.final_evaluation.metric_prefix == "in_distribution_test"
    assert cfg.final_evaluation.metrics_filename == "in_distribution_test.json"
    assert cfg.trainer.check_val_every_n_epoch == 1
    assert cfg.trainer.num_sanity_val_steps == 0
