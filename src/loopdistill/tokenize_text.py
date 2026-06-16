from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from loopdistill.data.text import iter_text_token_batches, load_text_dataset
from loopdistill.utils.logging import info, ok


def _cfg_get(cfg: DictConfig, key: str, default=None):
    return cfg.get(key, default) if cfg is not None else default


def _load_tokenizer(tokenizer_dir: str):
    tokenizer_path = Path(tokenizer_dir).expanduser() / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    try:
        return AutoTokenizer.from_pretrained(
            str(Path(tokenizer_dir).expanduser()),
            add_bos_token=False,
            add_eos_token=False,
        )
    except Exception:
        return PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            add_bos_token=False,
            add_eos_token=False,
        )


def _write_split(cfg: DictConfig, split_name: str, split_expr: str, num_samples: int) -> Path:
    tokenization = cfg.tokenization
    output_dir = Path(str(_cfg_get(tokenization, "output_dir"))).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split_name}.pt"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite token shard: {path}")

    tokenizer = _load_tokenizer(str(_cfg_get(tokenization, "tokenizer_dir")))
    dataset = load_text_dataset(
        dataset_path=_cfg_get(tokenization, "dataset_path", None),
        dataset_id=_cfg_get(tokenization, "dataset_id", None),
        dataset_config_name=_cfg_get(tokenization, "dataset_config_name", None),
        split=split_expr,
        cache_dir=_cfg_get(tokenization, "cache_dir", None),
    )
    batches = iter_text_token_batches(
        dataset=dataset,
        encode_text=lambda text: tokenizer.encode(text, add_special_tokens=False),
        text_column=str(_cfg_get(tokenization, "text_column", "text")),
        batch_size=int(_cfg_get(tokenization, "batch_size", 64)),
        seq_len=int(_cfg_get(tokenization, "seq_len", 512)),
        num_samples=int(num_samples),
        seed=int(cfg.seed) + int(_cfg_get(tokenization, "seed_offset", 0)),
        shuffle=bool(_cfg_get(tokenization, "shuffle", True)),
        max_text_chars=_cfg_get(tokenization, "max_text_chars", 200_000),
    )
    token_parts = []
    mask_parts = []
    for batch_idx, batch in enumerate(batches):
        if batch_idx % 10 == 0:
            info(f"Tokenizing {split_name}: batch {batch_idx}")
        token_parts.append(batch["tokens"].to(torch.int32))
        mask_parts.append(batch["attention_mask"].bool())
    if not token_parts:
        raise RuntimeError(f"Tokenization produced zero samples for split {split_name}.")
    tokens = torch.cat(token_parts, dim=0)[:num_samples]
    attention_mask = torch.cat(mask_parts, dim=0)[:num_samples]
    torch.save(
        {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "split": split_expr,
            "dataset_id": _cfg_get(tokenization, "dataset_id", None),
            "dataset_config_name": _cfg_get(tokenization, "dataset_config_name", None),
            "seq_len": int(tokens.shape[1]),
        },
        path,
    )
    ok(f"Wrote {tokens.shape[0]} token samples to {path}")
    return path


def run(cfg: DictConfig) -> None:
    tokenization = cfg.tokenization
    split_map = {
        "train": str(_cfg_get(tokenization, "train_split", "train[:98%]")),
        "val": str(_cfg_get(tokenization, "val_split", "train[98%:99%]")),
        "test": str(_cfg_get(tokenization, "test_split", "train[99%:]")),
    }
    sample_map = {
        "train": int(_cfg_get(tokenization, "train_samples", 8192)),
        "val": int(_cfg_get(tokenization, "val_samples", 512)),
        "test": int(_cfg_get(tokenization, "test_samples", 512)),
    }
    for split_name, split_expr in split_map.items():
        _write_split(cfg, split_name, split_expr, sample_map[split_name])


def main() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()

