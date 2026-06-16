from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch


def load_text_dataset(
    *,
    dataset_path: str | None = None,
    dataset_id: str | None = None,
    dataset_config_name: str | None = None,
    split: str = "train",
    cache_dir: str | None = None,
):
    from datasets import DatasetDict, load_dataset, load_from_disk

    if dataset_path:
        path = Path(dataset_path).expanduser()
        try:
            dataset = load_from_disk(str(path))
        except FileNotFoundError:
            if not dataset_id:
                raise
            dataset = load_dataset(dataset_id, dataset_config_name, split=split, cache_dir=cache_dir)
    elif dataset_id:
        dataset = load_dataset(dataset_id, dataset_config_name, split=split, cache_dir=cache_dir)
    else:
        raise ValueError("Text extraction requires either dataset_path or dataset_id.")

    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split {split!r} not found. Available splits: {sorted(dataset)}")
        return dataset[split]
    return dataset


def iter_text_token_batches(
    *,
    dataset: Any,
    encode_text: Callable[[str], list[int]],
    text_column: str = "text",
    batch_size: int = 4,
    seq_len: int = 256,
    num_samples: int = 1024,
    seed: int = 17,
    shuffle: bool = True,
    max_text_chars: int | None = 200_000,
    shard_index: int = 0,
    shard_count: int = 1,
) -> Iterator[dict[str, torch.Tensor]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if num_samples <= 0:
        return
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count.")

    if shuffle and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed)

    token_buffer: list[int] = []
    chunks: list[torch.Tensor] = []
    emitted = 0
    seen_chunks = 0
    target_samples = (num_samples + shard_count - 1) // shard_count

    for row in dataset:
        if emitted >= target_samples:
            break
        if text_column not in row:
            raise KeyError(f"Text column {text_column!r} not found in dataset row.")
        text = row[text_column]
        if not isinstance(text, str) or not text:
            continue
        if max_text_chars is not None:
            text = text[:max_text_chars]
        token_buffer.extend(int(token) for token in encode_text(text))

        while len(token_buffer) >= seq_len and emitted + len(chunks) < target_samples:
            chunk = token_buffer[:seq_len]
            del token_buffer[:seq_len]
            if seen_chunks % shard_count == shard_index:
                chunks.append(torch.tensor(chunk, dtype=torch.long))
            if len(chunks) == batch_size:
                yield _make_batch(chunks)
                emitted += len(chunks)
                chunks = []
            seen_chunks += 1

    if chunks and emitted < target_samples:
        yield _make_batch(chunks)


def _make_batch(chunks: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    tokens = torch.stack(chunks)
    return {
        "tokens": tokens,
        "attention_mask": torch.ones_like(tokens, dtype=torch.bool),
    }
