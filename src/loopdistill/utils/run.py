from __future__ import annotations

import json
import os
from pathlib import Path
from time import time
from typing import Any

from omegaconf import DictConfig, OmegaConf


def run_dirs(output_dir: str | os.PathLike[str]) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "root": root,
        "artifacts": root / "artifacts",
        "metrics": root / "metrics",
        "plots": root / "plots",
        "reports": root / "reports",
    }


def ensure_run_dirs(output_dir: str | os.PathLike[str]) -> dict[str, Path]:
    dirs = run_dirs(output_dir)
    root = dirs["root"]
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {root}. "
            "Choose a new run_id to avoid overwriting artifacts."
        )
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_resolved_config(cfg: DictConfig, output_dir: str | os.PathLike[str]) -> None:
    artifact_dir = Path(output_dir) / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=artifact_dir / "config_resolved.yaml", resolve=True)


def write_run_summary(
    output_dir: str | os.PathLike[str],
    *,
    title: str,
    metrics: dict[str, Any] | None = None,
    started_at: float | None = None,
) -> None:
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    elapsed = None if started_at is None else time() - started_at
    lines = [f"# {title}", ""]
    if elapsed is not None:
        lines += [f"- elapsed_seconds: {elapsed:.3f}"]
    if metrics:
        lines += ["", "## Metrics"]
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
    (report_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if metrics is not None:
        metrics_dir = Path(output_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "test.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
