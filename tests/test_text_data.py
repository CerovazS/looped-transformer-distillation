from __future__ import annotations

import pytest

from loopdistill.data.text import iter_text_token_batches
from loopdistill.teachers.attractor import AttractorTeacher


def test_iter_text_token_batches_chunks_fixed_length():
    dataset = [{"text": "abcdef"}, {"text": "ghijkl"}]

    batches = list(
        iter_text_token_batches(
            dataset=dataset,
            encode_text=lambda text: [ord(char) for char in text],
            batch_size=2,
            seq_len=3,
            num_samples=3,
            shuffle=False,
        )
    )

    assert len(batches) == 2
    assert batches[0]["tokens"].shape == (2, 3)
    assert batches[1]["tokens"].shape == (1, 3)
    assert batches[0]["attention_mask"].all()


def test_iter_text_token_batches_shards_are_disjoint():
    dataset = [{"text": "abcdefghijkl"}]
    common = {
        "dataset": dataset,
        "encode_text": lambda text: [ord(char) for char in text],
        "batch_size": 2,
        "seq_len": 2,
        "num_samples": 6,
        "shuffle": False,
        "shard_count": 2,
    }

    shard0 = list(iter_text_token_batches(**common, shard_index=0))
    shard1 = list(iter_text_token_batches(**common, shard_index=1))

    tokens0 = {tuple(row.tolist()) for batch in shard0 for row in batch["tokens"]}
    tokens1 = {tuple(row.tolist()) for batch in shard1 for row in batch["tokens"]}
    assert tokens0
    assert tokens1
    assert tokens0.isdisjoint(tokens1)


def test_attractor_teacher_requires_consecutive_depths():
    teacher = AttractorTeacher(
        repo_path="/missing",
        checkpoint_dir="/missing",
        device="cpu",
        dtype="float32",
        storage_dtype="float32",
    )

    with pytest.raises(ValueError, match="consecutive depths"):
        teacher._validate_depths([0, 2])
