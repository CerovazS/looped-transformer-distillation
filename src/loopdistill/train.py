from __future__ import annotations

import os
import sys
from pathlib import Path
from time import sleep, time

import hydra
import lightning as L
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from loopdistill.train_module import DistillationModule
from loopdistill.utils.logging import info, ok
from loopdistill.utils.run import ensure_run_dirs, run_dirs, save_resolved_config, write_run_summary


def _is_global_zero_env() -> bool:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ.get("LOCAL_RANK", "0")) == 0 and int(os.environ.get("NODE_RANK", "0")) == 0
    return int(os.environ.get("RANK", "0")) == 0


def _wait_for_run_dirs(output_dir: str, timeout_seconds: int = 120) -> None:
    root = Path(output_dir)
    for _ in range(timeout_seconds * 10):
        if (root / "artifacts" / "config_resolved.yaml").exists():
            return
        sleep(0.1)
    raise TimeoutError(f"Timed out waiting for rank 0 to initialize run directory: {output_dir}")


def _live_teacher_device(cfg: DictConfig) -> torch.device | None:
    trainer_cfg = cfg.get("trainer", {})
    accelerator = str(trainer_cfg.get("accelerator", "auto"))
    if accelerator not in {"gpu", "cuda", "auto"} or not torch.cuda.is_available():
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return torch.device(f"cuda:{local_rank}")


def run(cfg: DictConfig) -> None:
    started = time()
    if _is_global_zero_env():
        dirs = ensure_run_dirs(cfg.output_dir)
        save_resolved_config(cfg, cfg.output_dir)
    else:
        _wait_for_run_dirs(cfg.output_dir)
        dirs = run_dirs(cfg.output_dir)
    L.seed_everything(cfg.seed, workers=True)

    info("Instantiating data, student, and loss modules")
    data = instantiate(cfg.data)
    student = instantiate(cfg.student)
    loss_module = instantiate(cfg.loss)
    quality_evaluator = instantiate(cfg.eval_quality)
    live_cfg = cfg.get("live", {})
    final_eval_cfg = cfg.get("final_evaluation", {})
    teacher = instantiate(cfg.teacher) if bool(live_cfg.get("enabled", False)) else None
    module = DistillationModule(
        student=student,
        loss_module=loss_module,
        quality_evaluator=quality_evaluator,
        teacher=teacher,
        live_depths=live_cfg.get("depths", None),
        metrics_dir=str(dirs["metrics"]),
        test_metric_prefix=str(final_eval_cfg.get("metric_prefix", "test")),
    )
    module._connect_live_teacher_student(device=_live_teacher_device(cfg))
    trainer = instantiate(cfg.trainer)
    trainer.fit(module, datamodule=data)
    if bool(final_eval_cfg.get("enabled", True)):
        trainer.test(module, datamodule=data)
    if trainer.is_global_zero:
        write_run_summary(
            cfg.output_dir,
            title="LoopDistill training run",
            metrics={k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items() if hasattr(v, "detach")},
            started_at=started,
            metrics_filename=str(final_eval_cfg.get("metrics_filename", "test.json")),
        )
        ok(f"Training complete: {cfg.output_dir}")


def main() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
