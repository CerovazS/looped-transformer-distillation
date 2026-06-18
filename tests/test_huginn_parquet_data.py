from __future__ import annotations

import torch

from loopdistill.data.huginn_parquet import (
    _make_token_batch,
    _partition_files,
    _tokens_from_huginn_row,
)


def test_huginn_row_keeps_full_available_context_without_seq_len_cap():
    row = {"input_ids": [1, 2, 3, 4, 5]}

    tokens = _tokens_from_huginn_row(
        row,
        input_ids_column="input_ids",
        input_length=None,
        drop_last_token=True,
        pad_token_id=0,
    )

    assert tokens.tolist() == [1, 2, 3, 4]


def test_huginn_batch_pads_variable_rows():
    batch = _make_token_batch(
        [torch.tensor([1, 2, 3]), torch.tensor([4])],
        pad_token_id=0,
    )

    assert batch["tokens"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["attention_mask"].tolist() == [[True, True, True], [True, False, False]]


def test_huginn_file_partition_is_disjoint():
    files = [f"shard_{idx}.parquet" for idx in range(8)]

    train, val, test = _partition_files(files, val_file_count=2, test_file_count=1, seed=17)

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert len(train) == 5
