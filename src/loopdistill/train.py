from __future__ import annotations

import sys
from pathlib import Path
from time import time

import hydra
import lightning as L
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from loopdistill.train_module import DistillationModule
from loopdistill.utils.logging import info, ok
from loopdistill.utils.run import ensure_run_dirs, save_resolved_config, write_run_summary


def run(cfg: DictConfig) -> None:
    started = time()
    dirs = ensure_run_dirs(cfg.output_dir)
    save_resolved_config(cfg, cfg.output_dir)
    L.seed_everything(cfg.seed, workers=True)

    info("Instantiating data, student, and loss modules")
    data = instantiate(cfg.data)
    student = instantiate(cfg.student)
    loss_module = instantiate(cfg.loss)
    quality_evaluator = instantiate(cfg.eval_quality)
    live_cfg = cfg.get("live", {})
    teacher = instantiate(cfg.teacher) if bool(live_cfg.get("enabled", False)) else None
    module = DistillationModule(
        student=student,
        loss_module=loss_module,
        quality_evaluator=quality_evaluator,
        teacher=teacher,
        live_depths=live_cfg.get("depths", None),
        metrics_dir=str(dirs["metrics"]),
    )
    trainer = instantiate(cfg.trainer)
    trainer.fit(module, datamodule=data)
    trainer.test(module, datamodule=data)
    write_run_summary(
        cfg.output_dir,
        title="LoopDistill training run",
        metrics={k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items() if hasattr(v, "detach")},
        started_at=started,
    )
    ok(f"Training complete: {cfg.output_dir}")


def main() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
