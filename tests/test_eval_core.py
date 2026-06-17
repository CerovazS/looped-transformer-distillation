from __future__ import annotations

import torch
from omegaconf import OmegaConf

from loopdistill.evaluate_core import _batch_sequences, _kl_and_agreement, _render_prompts_lm, _render_prompts_mc, sequence_nll


class CharTokenizer:
    bos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(ch) % 50 for ch in text]


def test_sequence_nll_scores_only_continuation_tokens():
    logits = torch.full((1, 4, 5), -8.0)
    tokens = torch.tensor([[0, 1, 2, 3]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    prefix_lengths = torch.tensor([2])
    logits[0, 0, 4] = 8.0
    logits[0, 1, 2] = 8.0
    logits[0, 2, 3] = 8.0

    per_sequence, total, count = sequence_nll(logits, tokens, mask, prefix_lengths)

    assert count.item() == 2
    assert per_sequence.item() < 1e-5
    assert total.item() < 1e-5


def test_kl_and_agreement_identical_logits_are_perfect():
    logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    metrics = _kl_and_agreement(logits, logits.clone(), mask, temperature=1.0, top_k=1)

    kl_sum, kl_count = metrics["kl"]
    top1_sum, top1_count = metrics["top1"]
    assert kl_count.item() == 2
    assert kl_sum.item() < 1e-6
    assert top1_count.item() == 2
    assert top1_sum.item() == 2


def test_core_rendering_uses_fewshot_and_bos_span_indices():
    cfg = OmegaConf.create({"eval": {"prepend_bos": True}})
    tokenizer = CharTokenizer()
    item = {"query": "Q?", "choices": ["A", "B"], "gold": 1}
    fewshot = [{"query": "F?", "choices": ["X", "Y"], "gold": 0}]

    prompts = _render_prompts_mc(item, " ", fewshot)
    tokens, starts, ends = _batch_sequences(tokenizer, prompts, "multiple_choice", cfg)

    assert prompts[0].startswith("F? X\nQ? ")
    assert all(seq[0] == tokenizer.bos_token_id for seq in tokens)
    assert starts == [len(tokens[0]) - 1, len(tokens[1]) - 1]
    assert ends == [len(tokens[0]), len(tokens[1])]


def test_core_lm_renderer_splits_prompt_and_continuation():
    cfg = OmegaConf.create({"eval": {"prepend_bos": False}})
    tokenizer = CharTokenizer()
    item = {"context": " context  ", "continuation": "answer"}

    prompts = _render_prompts_lm(item, " ", [])
    tokens, starts, ends = _batch_sequences(tokenizer, prompts, "language_modeling", cfg)

    assert prompts == ["context", "context answer"]
    assert len(tokens) == 1
    assert starts[0] == len(tokenizer.encode("context"))
    assert ends[0] == len(tokenizer.encode("context answer"))
