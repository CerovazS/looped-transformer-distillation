from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from loopdistill.data.writer import TrajectoryShardWriter
from loopdistill.utils.logging import ok


def run(cfg: DictConfig) -> None:
    teacher = instantiate(cfg.teacher)
    output_dir = Path(cfg.paths.manifest_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing trajectory manifest: {manifest}. "
            "Choose a new paths.manifest_path or remove it explicitly."
        )

    batch_size = 16
    seq_len = 24
    vocab_size = int(getattr(teacher, "vocab_size", 128))
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    depths = list(range(int(getattr(teacher, "max_depth", 4)) + 1))
    output = teacher.run_batch(tokens=tokens, attention_mask=attention_mask, depths=depths)
    tensors = {
        "tokens": tokens.cpu(),
        "attention_mask": attention_mask.cpu(),
        "K": torch.full((batch_size,), output.z.shape[1] - 1, dtype=torch.long),
        "z": output.z.cpu(),
        "logits": None if output.logits is None else output.logits.cpu(),
        "loss_K": output.loss_K.cpu(),
        "residual_norm": output.residual_norm.cpu(),
        "solver_iters": output.solver_iters.cpu(),
    }
    tensors = {k: v for k, v in tensors.items() if v is not None}
    writer = TrajectoryShardWriter(
        output_dir,
        teacher_id=teacher.teacher_id,
        teacher_repo=teacher.__class__.__module__,
        dataset_id="synthetic-smoke",
        split="train",
    )
    writer.write_shard(tensors, shard_name="shard_00000.pt", metadata={"source": "extract_trajectories"})
    ok(f"Wrote trajectory manifest: {writer.manifest_path}")


def main() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
