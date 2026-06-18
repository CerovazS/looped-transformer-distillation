from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def _resolve_parquet_files(dataset_dir: str | Path, split_glob: str = "*.parquet") -> list[str]:
    root = Path(dataset_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Huginn parquet dataset_dir not found: {root}")
    files = sorted(str(path) for path in root.glob(split_glob))
    if not files:
        files = sorted(str(path) for path in root.glob(f"**/{split_glob}"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root} with glob {split_glob!r}.")
    return files


def _partition_files(
    files: Sequence[str],
    *,
    val_file_count: int,
    test_file_count: int,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    if val_file_count < 0 or test_file_count < 0:
        raise ValueError("val_file_count and test_file_count must be non-negative.")
    if val_file_count + test_file_count >= len(files):
        raise ValueError(
            "Huginn parquet split needs at least one train file after val/test reservation. "
            f"Got {len(files)} files, val_file_count={val_file_count}, test_file_count={test_file_count}."
        )
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(files), generator=generator).tolist()
    shuffled = [files[index] for index in order]
    val = shuffled[:val_file_count]
    test = shuffled[val_file_count : val_file_count + test_file_count]
    train = shuffled[val_file_count + test_file_count :]
    return train, val, test


def _tokens_from_huginn_row(
    row: dict[str, Any],
    *,
    input_ids_column: str,
    input_length: int | None,
    drop_last_token: bool,
    pad_token_id: int,
) -> torch.Tensor:
    if input_ids_column not in row:
        raise KeyError(f"Column {input_ids_column!r} not found in Huginn parquet row.")
    ids = row[input_ids_column]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    ids = [int(token) for token in ids]
    if drop_last_token:
        ids = ids[:-1]
    if input_length is not None:
        if input_length <= 0:
            raise ValueError("input_length must be positive when set.")
        ids = ids[:input_length]
        if len(ids) < input_length:
            ids.extend([pad_token_id] * (input_length - len(ids)))
    if not ids:
        raise ValueError("Huginn parquet row produced an empty token sequence.")
    return torch.tensor(ids, dtype=torch.long)


def _make_token_batch(chunks: list[torch.Tensor], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(int(chunk.shape[0]) for chunk in chunks)
    tokens = torch.full((len(chunks), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(chunks), max_len), dtype=torch.bool)
    for index, chunk in enumerate(chunks):
        tokens[index, : chunk.shape[0]] = chunk
        attention_mask[index, : chunk.shape[0]] = True
    return {"tokens": tokens, "attention_mask": attention_mask}


class HuginnParquetIterableDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Token batches from Huginn's already-tokenized parquet rows."""

    def __init__(
        self,
        *,
        data_files: Sequence[str],
        input_ids_column: str = "input_ids",
        batch_size: int = 1,
        input_length: int | None = None,
        drop_last_token: bool = True,
        pad_token_id: int = 65509,
        num_samples: int = 1024,
        seed: int = 17,
        shuffle_files: bool = True,
    ):
        self.data_files = list(data_files)
        self.input_ids_column = input_ids_column
        self.batch_size = int(batch_size)
        self.input_length = input_length
        self.drop_last_token = bool(drop_last_token)
        self.pad_token_id = int(pad_token_id)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.shuffle_files = bool(shuffle_files)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from datasets import load_dataset

        data_files = list(self.data_files)
        if self.shuffle_files:
            generator = torch.Generator().manual_seed(self.seed)
            order = torch.randperm(len(data_files), generator=generator).tolist()
            data_files = [data_files[index] for index in order]
        dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
        shard_index, shard_count = self._distributed_shard()
        chunks: list[torch.Tensor] = []
        emitted = 0
        seen = 0
        target_samples = (self.num_samples + shard_count - 1) // shard_count
        for row in dataset:
            if emitted >= target_samples:
                break
            if seen % shard_count == shard_index:
                chunks.append(
                    _tokens_from_huginn_row(
                        row,
                        input_ids_column=self.input_ids_column,
                        input_length=self.input_length,
                        drop_last_token=self.drop_last_token,
                        pad_token_id=self.pad_token_id,
                    )
                )
                if len(chunks) == self.batch_size:
                    yield _make_token_batch(chunks, self.pad_token_id)
                    emitted += len(chunks)
                    chunks = []
            seen += 1
        if chunks and emitted < target_samples:
            yield _make_token_batch(chunks, self.pad_token_id)

    def _distributed_shard(self) -> tuple[int, int]:
        rank = 0
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        worker = get_worker_info()
        if worker is None:
            return rank, world_size
        return rank * worker.num_workers + worker.id, world_size * worker.num_workers


class HuginnParquetDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        dataset_dir: str,
        split_glob: str = "*.parquet",
        input_ids_column: str = "input_ids",
        batch_size: int = 1,
        input_length: int | None = None,
        drop_last_token: bool = True,
        pad_token_id: int = 65509,
        train_samples: int = 1024,
        val_samples: int = 64,
        test_samples: int = 64,
        val_file_count: int = 2,
        test_file_count: int = 2,
        seed: int = 17,
        shuffle_files: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        files = _resolve_parquet_files(dataset_dir, split_glob)
        self.train_files, self.val_files, self.test_files = _partition_files(
            files,
            val_file_count=val_file_count,
            test_file_count=test_file_count,
            seed=seed,
        )
        self.common = {
            "input_ids_column": input_ids_column,
            "batch_size": batch_size,
            "input_length": input_length,
            "drop_last_token": drop_last_token,
            "pad_token_id": pad_token_id,
            "shuffle_files": shuffle_files,
        }
        self.train_samples = int(train_samples)
        self.val_samples = int(val_samples)
        self.test_samples = int(test_samples)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._dataset(self.train_files, self.train_samples, seed_offset=0)
            self.val_dataset = self._dataset(self.val_files, self.val_samples, seed_offset=10_000)
        if stage in (None, "test"):
            self.test_dataset = self._dataset(self.test_files, self.test_samples, seed_offset=20_000)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)

    def _dataset(
        self,
        data_files: Sequence[str],
        num_samples: int,
        *,
        seed_offset: int,
    ) -> HuginnParquetIterableDataset:
        return HuginnParquetIterableDataset(
            data_files=data_files,
            num_samples=num_samples,
            seed=self.seed + seed_offset,
            **self.common,
        )

    def _loader(self, dataset: HuginnParquetIterableDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
