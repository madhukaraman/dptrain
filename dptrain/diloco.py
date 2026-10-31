"""DiLoCo: inner AdamW on each peer, Nesterov SGD on the outer pseudo-gradients.

Pseudo-gradient for peer k is (theta_outer - theta_k) after H inner steps, i.e.
the net displacement that peer's local training produced. Those are averaged and
fed to the outer optimizer as if they were a gradient.

Transport is int8 with error feedback: the quantization residual is carried
forward and added to the next round's payload, so compression error is deferred
rather than discarded.
"""

import copy
from dataclasses import dataclass, field

import torch

from .data import get_batch


def quantize_int8(t: torch.Tensor):
    """Symmetric per-tensor int8. Returns (codes, scale)."""
    scale = t.abs().max() / 127.0
    if scale == 0:
        return torch.zeros_like(t, dtype=torch.int8), torch.tensor(1.0)
    return torch.round(t / scale).clamp(-127, 127).to(torch.int8), scale


def dequantize_int8(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return codes.to(torch.float32) * scale


@dataclass
class Peer:
    peer_id: int
    shard: torch.Tensor
    model: torch.nn.Module
    opt: torch.optim.Optimizer
    error: dict = field(default_factory=dict)   # error-feedback buffers
    joined_at: int = 0
    alive: bool = True

    def inner_steps(self, outer_state: dict, steps: int, batch_size: int,
                    block_size: int, rng: torch.Generator, device: str) -> float:
        """Reset to the broadcast weights, train locally, return mean loss."""
        self.model.load_state_dict(outer_state)
        self.model.train()
        total = 0.0
        for _ in range(steps):
            x, y = get_batch(self.shard, batch_size, block_size, rng, device)
            _, loss = self.model(x, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total += loss.item()
        return total / steps

    def pseudo_gradient(self, outer_state: dict, compress: bool):
        """Displacement produced by local training, optionally int8-compressed.
        Returns (pseudo_grad_dict, bytes_on_wire)."""
        local = self.model.state_dict()
        payload, wire_bytes = {}, 0
        for name, theta in outer_state.items():
            delta = theta - local[name]
            if not compress:
                payload[name] = delta
                wire_bytes += delta.numel() * 4
                continue
            corrected = delta + self.error.get(name, torch.zeros_like(delta))
            codes, scale = quantize_int8(corrected)
            recovered = dequantize_int8(codes, scale)
            self.error[name] = corrected - recovered
            payload[name] = recovered
            wire_bytes += codes.numel() + 4  # int8 payload + fp32 scale
        return payload, wire_bytes


class OuterOptimizer:
    """Nesterov momentum over averaged pseudo-gradients (DiLoCo's outer loop)."""

    def __init__(self, state: dict, lr: float = 0.7, momentum: float = 0.9):
        self.state = {k: v.clone() for k, v in state.items()}
        self.lr, self.momentum = lr, momentum
        self.buf = {k: torch.zeros_like(v) for k, v in state.items()}

    def step(self, pseudo_grads: list):
        if not pseudo_grads:
            return  # every peer dropped this round; outer weights stand
        for name in self.state:
            avg = torch.stack([g[name] for g in pseudo_grads]).mean(dim=0)
            self.buf[name].mul_(self.momentum).add_(avg)
            self.state[name] -= self.lr * (avg + self.momentum * self.buf[name])


def make_peer(peer_id: int, shard: torch.Tensor, cfg, inner_lr: float,
              device: str, round_idx: int = 0) -> Peer:
    from .model import TinyGPT
    model = TinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=inner_lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    return Peer(peer_id, shard, model, opt, joined_at=round_idx)


@torch.no_grad()
def evaluate(model, data, batch_size, block_size, device, iters=20, seed=1234):
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, batch_size, block_size, rng, device)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / iters


def clone_state(model) -> dict:
    return copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
