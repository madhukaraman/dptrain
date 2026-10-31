"""Synthetic multi-dialect corpus.

Each dialect has its own vocabulary of invented words and its own bigram
transitions over them. That gives us two things a real corpus would:
learnable structure (loss drops well below uniform), and a principled way to
build non-IID shards -- peer k sees mostly dialect k, which is the regime
DiLoCo claims robustness to.
"""

import string

import torch

ALPHABET = string.ascii_lowercase + " ."
STOI = {ch: i for i, ch in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET)


def _make_dialect(rng: torch.Generator, n_words: int = 24) -> dict:
    words = []
    for _ in range(n_words):
        length = int(torch.randint(3, 7, (1,), generator=rng).item())
        idx = torch.randint(0, 26, (length,), generator=rng)
        words.append("".join(string.ascii_lowercase[i] for i in idx))
    # Sparse transition matrix: each word is followed by one of ~3 successors.
    trans = torch.zeros(n_words, n_words)
    for i in range(n_words):
        succ = torch.randperm(n_words, generator=rng)[:3]
        trans[i, succ] = torch.rand(3, generator=rng)
    trans /= trans.sum(dim=1, keepdim=True)
    return {"words": words, "trans": trans}


def build_dialects(n_dialects: int, seed: int = 0) -> list:
    rng = torch.Generator().manual_seed(seed)
    return [_make_dialect(rng) for _ in range(n_dialects)]


def sample_text(dialect: dict, n_chars: int, rng: torch.Generator) -> str:
    words, trans = dialect["words"], dialect["trans"]
    out, cur = [], 0
    while sum(len(w) + 1 for w in out) < n_chars:
        out.append(words[cur])
        cur = int(torch.multinomial(trans[cur], 1, generator=rng).item())
    return " ".join(out)[:n_chars] + "."


def encode(text: str) -> torch.Tensor:
    return torch.tensor([STOI[c] for c in text if c in STOI], dtype=torch.long)


def build_shards(n_peers: int, chars_per_peer: int, non_iid: bool, seed: int = 0):
    """One shard per peer. Under non_iid, peer k draws 85% of its text from
    dialect k; otherwise every peer sees a uniform mixture."""
    dialects = build_dialects(n_peers, seed=seed)
    rng = torch.Generator().manual_seed(seed + 1)
    shards = []
    for k in range(n_peers):
        parts = []
        if non_iid:
            parts.append(sample_text(dialects[k], int(chars_per_peer * 0.85), rng))
            others = [d for i, d in enumerate(dialects) if i != k]
            per_other = int(chars_per_peer * 0.15 / max(1, len(others)))
            parts += [sample_text(d, per_other, rng) for d in others]
        else:
            per = chars_per_peer // n_peers
            parts = [sample_text(d, per, rng) for d in dialects]
        shards.append(encode(" ".join(parts)))
    held_out = encode(" ".join(sample_text(d, 4000, rng) for d in dialects))
    return shards, held_out


def get_batch(data: torch.Tensor, batch_size: int, block_size: int,
              rng: torch.Generator, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=rng)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
