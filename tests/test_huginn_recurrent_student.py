from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from loopdistill.models.huginn import HuginnRecurrentStudent


class _FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(16, 4)
        self.prelude = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.adapter = nn.Linear(8, 4)
        self.core_block = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.coda = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.ln_f = nn.LayerNorm(4)


class _FakeHuginn(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            n_layers_in_prelude=2,
            n_layers_in_recurrent_block=4,
        )
        self.transformer = _FakeTransformer()
        self.lm_head = nn.Linear(4, 16)
        self.register_buffer("freqs_cis", torch.zeros(1, 32, 1))
        self.seen_block_idx: list[int] = []

    def embed_inputs(self, tokens, attention_mask=None):
        embeds = self.transformer.wte(tokens)
        return embeds, torch.tensor(1, device=tokens.device)

    def core_block_forward(
        self,
        x,
        input_embeds,
        freqs_cis,
        mask,
        past_key_values,
        block_idx,
        current_step,
    ):
        self.seen_block_idx.append(int(block_idx))
        y = self.transformer.adapter(torch.cat([x, input_embeds], dim=-1))
        for block in self.transformer.core_block:
            y = block(y)
        return y, block_idx + 4

    def predict_from_latents(self, latents):
        return SimpleNamespace(logits=self.lm_head(latents))


def test_huginn_recurrent_student_uses_huginn_core_block_and_step_indexing():
    model = _FakeHuginn()
    student = HuginnRecurrentStudent.__new__(HuginnRecurrentStudent)
    nn.Module.__init__(student)
    student.model = model
    student.target_depth = 16
    student.return_logits = False

    z = torch.randn(2, 3, 4)
    tokens = torch.randint(0, 16, (2, 3))
    t = torch.tensor([0.0, 0.5])
    delta = torch.full((2,), 0.25)

    out = student(z, t, delta, tokens=tokens)

    assert out.z_next is not None
    assert out.z_next.shape == z.shape
    assert model.seen_block_idx == [1, 33]
