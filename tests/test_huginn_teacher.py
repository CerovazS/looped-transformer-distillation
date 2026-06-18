from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from loopdistill.teachers.huginn import HuginnTeacher


class FakeHuginnModel:
    config = SimpleNamespace(n_embd=3, padded_vocab_size=7, vocab_size=7)

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def embed_inputs(self, input_ids, attention_mask=None):
        embeds = input_ids.float().unsqueeze(-1).repeat(1, 1, self.config.n_embd)
        return embeds, torch.tensor(-1)

    def initialize_state(self, input_embeds, scale=1.0):
        return torch.zeros_like(input_embeds) + scale

    def iterate_one_step(
        self,
        input_embeds,
        input_states,
        block_idx,
        attention_mask=None,
        past_key_values=None,
        current_step=0,
    ):
        return input_states + input_embeds + float(current_step), block_idx + 1, current_step + 1

    def predict_from_latents(self, latents):
        base = latents.sum(dim=-1, keepdim=True)
        logits = base.repeat(1, 1, self.config.padded_vocab_size)
        return SimpleNamespace(logits=logits)


def test_huginn_teacher_collects_consecutive_recurrent_states():
    teacher = HuginnTeacher(device="cpu", dtype="float32", storage_dtype="float32", return_logits=True)
    teacher._model = FakeHuginnModel()
    tokens = torch.tensor([[2, 3]])
    mask = torch.ones_like(tokens, dtype=torch.bool)

    output = teacher.run_batch(tokens=tokens, attention_mask=mask, depths=[0, 1, 2])

    assert output.z.shape == (1, 3, 2, 3)
    assert torch.allclose(output.z[:, 0], torch.ones(1, 2, 3))
    assert torch.allclose(output.z[:, 1], torch.tensor([[[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]]))
    assert torch.allclose(output.z[:, 2], torch.tensor([[[6.0, 6.0, 6.0], [8.0, 8.0, 8.0]]]))
    assert output.logits.shape == (1, 1, 2, 7)
    assert output.solver_iters.item() == 2


def test_huginn_teacher_requires_consecutive_depths():
    teacher = HuginnTeacher(device="cpu")

    with pytest.raises(ValueError, match="consecutive depths"):
        teacher._validate_depths([0, 2])
