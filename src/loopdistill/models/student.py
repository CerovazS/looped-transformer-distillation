from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class StudentOutput:
    velocity: torch.Tensor
    z_next: torch.Tensor | None = None
    avg_velocity: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class ScalarEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(-1, 1).float())


class StudentFlowModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        vocab_size: int | None = None,
        max_seq_len: int = 2048,
        use_token_context: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.use_token_context = use_token_context and vocab_size is not None
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim) if self.use_token_context else None
        self.t_embed = ScalarEmbedding(hidden_dim)
        self.delta_embed = ScalarEmbedding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.velocity_head = nn.Linear(hidden_dim, latent_dim)
        self.flow_head = nn.Linear(hidden_dim, latent_dim)
        self.avg_velocity_head = nn.Linear(hidden_dim, latent_dim)
        self.logit_head = nn.Linear(hidden_dim, vocab_size) if vocab_size is not None else None

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        delta: torch.Tensor,
        tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        mode: str = "velocity",
    ) -> StudentOutput:
        batch, seq_len, _ = z_t.shape
        pos = torch.arange(seq_len, device=z_t.device).unsqueeze(0).expand(batch, seq_len)
        h = self.in_proj(z_t) + self.pos_embed(pos)
        h = h + self.t_embed(t).unsqueeze(1) + self.delta_embed(delta).unsqueeze(1)
        if self.token_embed is not None and tokens is not None:
            h = h + self.token_embed(tokens.clamp_min(0))
        if context is not None:
            h = h + context
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        velocity = self.velocity_head(h)
        avg_velocity = self.avg_velocity_head(h)
        z_next = z_t + delta.reshape(batch, 1, 1).to(z_t.dtype) * self.flow_head(h)
        logits = None if self.logit_head is None else self.logit_head(h)
        if mode == "flow_map":
            return StudentOutput(velocity=velocity, z_next=z_next, avg_velocity=avg_velocity, logits=logits)
        return StudentOutput(velocity=velocity, z_next=z_next, avg_velocity=avg_velocity, logits=logits)
