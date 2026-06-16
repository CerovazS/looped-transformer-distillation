from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from loopdistill.data.text import iter_text_token_batches, load_text_dataset
from loopdistill.data.writer import TrajectoryShardWriter
from loopdistill.utils.logging import info, ok


def _cfg_get(cfg: DictConfig, key: str, default=None):
    return cfg.get(key, default) if cfg is not None else default


def _depths_for(cfg: DictConfig, teacher) -> list[int]:
    configured = _cfg_get(cfg.extraction, "depths", None)
    if configured is not None:
        return [int(depth) for depth in configured]
    max_depth = int(getattr(teacher, "max_depth", 4))
    return list(range(max_depth + 1))


def _iter_synthetic_batches(cfg: DictConfig, teacher):
    batch_size = int(_cfg_get(cfg.extraction, "batch_size", 16))
    seq_len = int(_cfg_get(cfg.extraction, "seq_len", 24))
    num_samples = int(_cfg_get(cfg.extraction, "num_samples", batch_size))
    vocab_size = int(getattr(teacher, "vocab_size", None) or 128)
    generator = torch.Generator().manual_seed(int(cfg.seed))
    emitted = 0
    while emitted < num_samples:
        current = min(batch_size, num_samples - emitted)
        tokens = torch.randint(1, vocab_size, (current, seq_len), generator=generator)
        attention_mask = torch.ones(current, seq_len, dtype=torch.bool)
        yield {"tokens": tokens, "attention_mask": attention_mask}
        emitted += current


def _iter_text_batches(cfg: DictConfig, teacher):
    extraction = cfg.extraction
    dataset = load_text_dataset(
        dataset_path=_cfg_get(extraction, "dataset_path", None),
        dataset_id=_cfg_get(extraction, "dataset_id", None),
        split=str(_cfg_get(extraction, "split", "train")),
        cache_dir=_cfg_get(extraction, "cache_dir", None),
    )
    if not hasattr(teacher, "encode_text"):
        raise TypeError(
            f"{teacher.__class__.__name__} does not expose encode_text(); "
            "text extraction requires a teacher tokenizer."
        )
    return iter_text_token_batches(
        dataset=dataset,
        encode_text=teacher.encode_text,
        text_column=str(_cfg_get(extraction, "text_column", "text")),
        batch_size=int(_cfg_get(extraction, "batch_size", 4)),
        seq_len=int(_cfg_get(extraction, "seq_len", 256)),
        num_samples=int(_cfg_get(extraction, "num_samples", 1024)),
        seed=int(cfg.seed),
        shuffle=bool(_cfg_get(extraction, "shuffle", True)),
        max_text_chars=_cfg_get(extraction, "max_text_chars", 200_000),
    )


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

    depths = _depths_for(cfg, teacher)
    source = str(_cfg_get(cfg.extraction, "source", "synthetic"))
    batches = _iter_synthetic_batches(cfg, teacher) if source == "synthetic" else _iter_text_batches(cfg, teacher)

    writer = TrajectoryShardWriter(
        output_dir,
        teacher_id=teacher.teacher_id,
        teacher_repo=teacher.__class__.__module__,
        teacher_ckpt=(
            None
            if getattr(teacher, "checkpoint_dir", None) is None
            else str(getattr(teacher, "checkpoint_dir"))
        ),
        tokenizer_id=getattr(teacher, "tokenizer_id", None),
        dataset_id=str(_cfg_get(cfg.extraction, "dataset_id", source)),
        split=str(_cfg_get(cfg.extraction, "split", "train")),
    )

    shard_count = 0
    sample_count = 0
    for shard_count, batch in enumerate(batches):
        tokens = batch["tokens"]
        attention_mask = batch["attention_mask"]
        info(f"Extracting shard {shard_count:05d} with batch size {tokens.shape[0]}")
        output = teacher.run_batch(tokens=tokens, attention_mask=attention_mask, depths=depths)
        current = int(tokens.shape[0])
        tensors = {
            "tokens": tokens.cpu(),
            "attention_mask": attention_mask.cpu(),
            "K": torch.full((current,), output.z.shape[1] - 1, dtype=torch.long),
            "z": output.z.cpu(),
            "logits": None if output.logits is None else output.logits.cpu(),
            "loss_K": output.loss_K.cpu(),
            "residual_norm": output.residual_norm.cpu(),
            "solver_iters": output.solver_iters.cpu(),
        }
        tensors = {key: value for key, value in tensors.items() if value is not None}
        writer.write_shard(
            tensors,
            shard_name=f"shard_{shard_count:05d}.pt",
            metadata={
                "source": source,
                "depths": depths,
                "seq_len": int(tokens.shape[1]),
            },
        )
        sample_count += current
    if sample_count == 0:
        raise RuntimeError("Trajectory extraction produced zero samples.")
    ok(f"Wrote trajectory manifest: {writer.manifest_path}")
    ok(f"Wrote {sample_count} samples across {shard_count + 1} shard(s)")


def main() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
