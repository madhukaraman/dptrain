"""A small causal transformer. Deliberately tiny: the point is protocol behaviour,
not model quality, so every run finishes in minutes on a laptop CPU."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    block_size: int = 64

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False),
        )

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        shape = (B, T, self.cfg.n_head, self.cfg.d_head)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Deliberately untied: the outer state is round-tripped as a flat dict,
        # and two keys aliasing one Parameter silently corrupts it.
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self) -> int:
        """Forward+backward FLOPs per token, the standard 6N approximation
        plus the attention term that 6N omits."""
        c = self.cfg
        n = self.n_params()
        return 6 * n + 12 * c.n_layer * c.d_model * c.block_size
