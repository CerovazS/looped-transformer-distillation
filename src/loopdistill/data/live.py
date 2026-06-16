from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from loopdistill.data.text import iter_text_token_batches, load_text_dataset


class TextTokenIterableDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        dataset_path: str | None = None,
        dataset_id: str | None = None,
        dataset_config_name: str | None = None,
        split: str = "train",
        cache_dir: str | None = None,
        tokenizer_dir: str,
        text_column: str = "text",
        batch_size: int = 2,
        seq_len: int = 128,
        num_samples: int = 1024,
        seed: int = 17,
        shuffle: bool = True,
        max_text_chars: int | None = 200_000,
    ):
        self.dataset_path = dataset_path
        self.dataset_id = dataset_id
        self.dataset_config_name = dataset_config_name
        self.split = split
        self.cache_dir = cache_dir
        self.tokenizer_dir = str(Path(tokenizer_dir).expanduser())
        self.text_column = text_column
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.seed = seed
        self.shuffle = shuffle
        self.max_text_chars = max_text_chars

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        dataset = load_text_dataset(
            dataset_path=self.dataset_path,
            dataset_id=self.dataset_id,
            dataset_config_name=self.dataset_config_name,
            split=self.split,
            cache_dir=self.cache_dir,
        )
        tokenizer = self._load_tokenizer()
        shard_index, shard_count = self._distributed_shard()
        yield from iter_text_token_batches(
            dataset=dataset,
            encode_text=lambda text: tokenizer.encode(text, add_special_tokens=False),
            text_column=self.text_column,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            num_samples=self.num_samples,
            seed=self.seed,
            shuffle=self.shuffle,
            max_text_chars=self.max_text_chars,
            shard_index=shard_index,
            shard_count=shard_count,
        )

    def _load_tokenizer(self):
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        tokenizer_path = Path(self.tokenizer_dir) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        try:
            return AutoTokenizer.from_pretrained(
                self.tokenizer_dir,
                add_bos_token=False,
                add_eos_token=False,
            )
        except Exception:
            return PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_path),
                add_bos_token=False,
                add_eos_token=False,
            )

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


class TextTokenDataModule(L.LightningDataModule):
    """Token-only datamodule for live teacher distillation."""

    def __init__(
        self,
        *,
        dataset_path: str | None = None,
        dataset_id: str | None = None,
        dataset_config_name: str | None = None,
        train_split: str = "train",
        val_split: str | None = None,
        test_split: str | None = None,
        cache_dir: str | None = None,
        tokenizer_dir: str,
        text_column: str = "text",
        batch_size: int = 2,
        seq_len: int = 128,
        train_samples: int = 1024,
        val_samples: int = 128,
        test_samples: int = 128,
        seed: int = 17,
        shuffle: bool = True,
        max_text_chars: int | None = 200_000,
        allow_train_reuse_for_smoke: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.kwargs: dict[str, Any] = {
            "dataset_path": dataset_path,
            "dataset_id": dataset_id,
            "dataset_config_name": dataset_config_name,
            "cache_dir": cache_dir,
            "tokenizer_dir": tokenizer_dir,
            "text_column": text_column,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "seed": seed,
            "shuffle": shuffle,
            "max_text_chars": max_text_chars,
        }
        self.train_split = train_split
        self.val_split = val_split or train_split
        self.test_split = test_split or self.val_split
        self.allow_train_reuse_for_smoke = bool(allow_train_reuse_for_smoke)
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str | None = None) -> None:
        self._validate_splits()
        if stage in (None, "fit"):
            self.train_dataset = self._dataset(self.train_split, self.train_samples, seed_offset=0)
            self.val_dataset = self._dataset(self.val_split, self.val_samples, seed_offset=10_000)
        if stage in (None, "test"):
            self.test_dataset = self._dataset(self.test_split, self.test_samples, seed_offset=20_000)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)

    def _dataset(
        self,
        split: str,
        num_samples: int,
        *,
        seed_offset: int,
    ) -> TextTokenIterableDataset:
        kwargs = dict(self.kwargs)
        kwargs["split"] = split
        kwargs["num_samples"] = num_samples
        kwargs["seed"] = int(kwargs["seed"]) + seed_offset
        return TextTokenIterableDataset(**kwargs)

    def _loader(self, dataset: TextTokenIterableDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _validate_splits(self) -> None:
        if self.allow_train_reuse_for_smoke:
            return
        splits = {self.train_split, self.val_split, self.test_split}
        if len(splits) < 3:
            raise ValueError(
                "TextTokenDataModule requires distinct train/val/test split expressions. "
                "For FineWeb-Edu sample-10BT use HF slices such as "
                "'train[:98%]', 'train[98%:99%]', 'train[99%:]'. "
                "Set allow_train_reuse_for_smoke=true only for explicit smoke tests."
            )
