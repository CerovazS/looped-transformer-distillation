from __future__ import annotations

import torch
import lightning as L
from hydra import compose, initialize_config_dir
from pathlib import Path
from torch.utils.data import DataLoader

from loopdistill.losses.p0 import LoopDistillationLoss
from loopdistill.models.student import StudentFlowModel
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


def test_hydra_live_experiment_overrides_defaults():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=blackwell_live_attractor140"])

    assert cfg.live.enabled is True
    assert cfg.data._target_ == "loopdistill.data.live.TextTokenDataModule"
    assert cfg.teacher._target_ == "loopdistill.teachers.attractor.AttractorTeacher"
    assert cfg.teacher.return_logits is True
    assert cfg.eval_quality.enabled is True
    assert cfg.output_dir.startswith("outputs/live_distill/")
