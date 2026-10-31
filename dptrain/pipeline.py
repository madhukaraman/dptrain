"""Pipeline parallelism across peers: the sub-replica path.

DiLoCo needs every peer to hold a full replica, so the hardware floor is "can
you train the whole model alone". Pipeline parallelism is the only way around
that -- each peer owns a contiguous slice of layers and passes activations to
the next.

The cost moves from one pseudo-gradient per round to two tensors per
micro-batch per boundary (activations forward, their gradients back). Whether
that trade is worth it depends on a ratio this module measures rather than
assumes: compute per stage scales with d_model squared, boundary traffic only
with d_model.

Stages run sequentially on one device. That is fine -- we are counting bytes
and FLOPs, not timing a laptop.
"""

from dataclasses import dataclass, field

import torch

from .diloco import dequantize_int8, quantize_int8


@dataclass
class Boundary:
    """Bytes crossing one stage boundary, counted from the real tensors."""
    fwd_bytes: int = 0
    bwd_bytes: int = 0

    @property
    def total(self) -> int:
        return self.fwd_bytes + self.bwd_bytes


@dataclass
class PipelineStats:
    boundaries: list = field(default_factory=list)
    stage_params: list = field(default_factory=list)
    micro_batches: int = 0

    @property
    def bytes_per_microbatch(self) -> int:
        n = max(1, self.micro_batches)
        return sum(b.total for b in self.boundaries) // n

    @property
    def peak_stage_params(self) -> int:
        return max(self.stage_params) if self.stage_params else 0


def partition(n_layers: int, n_stages: int) -> list:
    """Contiguous layer ranges, remainder spread over the earliest stages."""
    if n_stages > n_layers:
        raise ValueError(f"{n_stages} stages for {n_layers} layers")
    base, extra = divmod(n_layers, n_stages)
    bounds, start = [], 0
    for s in range(n_stages):
        size = base + (1 if s < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def _cross(t: torch.Tensor, compress: bool):
    """Move a tensor across a boundary, returning (tensor, bytes_on_wire).

    int8 here is lossy in a way int8 pseudo-gradients were not: there is no
    error-feedback buffer to carry the residual, because an activation is
    consumed once and never revisited.
    """
    if not compress:
        return t, t.numel() * t.element_size()
    codes, scale = quantize_int8(t)
    return dequantize_int8(codes, scale).view_as(t), codes.numel() + 4


def pipeline_step(model, idx, targets, n_stages: int, compress: bool = False,
                  stats: PipelineStats = None):
    """One forward+backward with real stage boundaries.

    Activations are detached at each boundary, which severs the autograd graph
    exactly as a machine boundary would. The backward pass therefore has to be
    walked one stage at a time, seeding each segment with the gradient that
    arrived from the stage above -- a single loss.backward() would stop at the
    last boundary and silently leave earlier stages ungradiented.

    Gradients come out identical to monolithic autograd; verify_equivalence
    asserts it.
    """
    cfg = model.cfg
    bounds = partition(cfg.n_layer, n_stages)
    stats = stats or PipelineStats()
    if not stats.boundaries:
        stats.boundaries = [Boundary() for _ in range(max(0, n_stages - 1))]
        stats.stage_params = _stage_param_counts(model, bounds)
    stats.micro_batches += 1

    T = idx.shape[1]
    h = model.tok(idx) + model.pos(torch.arange(T, device=idx.device))

    stage_in, stage_out = [], []
    for s, (lo, hi) in enumerate(bounds):
        if s > 0:
            sent, nbytes = _cross(h.detach(), compress)
            stats.boundaries[s - 1].fwd_bytes += nbytes
            h = sent.requires_grad_(True)
        stage_in.append(h)
        for blk in model.blocks[lo:hi]:
            h = blk(h)
        stage_out.append(h)

    logits = model.head(model.ln_f(stage_out[-1]))
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.reshape(-1))

    # Seeds the last stage only -- the graph is cut at every boundary.
    loss.backward()

    for s in range(n_stages - 1, 0, -1):
        g = stage_in[s].grad
        if g is None:
            break
        sent, nbytes = _cross(g, compress)
        stats.boundaries[s - 1].bwd_bytes += nbytes
        stage_out[s - 1].backward(sent)

    return loss, stats


def _stage_param_counts(model, bounds) -> list:
    counts = []
    for s, (lo, hi) in enumerate(bounds):
        n = sum(p.numel() for blk in model.blocks[lo:hi] for p in blk.parameters())
        if s == 0:
            n += model.tok.weight.numel() + model.pos.weight.numel()
        if s == len(bounds) - 1:
            n += model.head.weight.numel() + sum(p.numel() for p in model.ln_f.parameters())
        counts.append(n)
    return counts


@torch.no_grad()
def _flat_grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def verify_equivalence(model, idx, targets, n_stages: int, tol: float = 1e-5):
    """Pipelined execution must produce the same gradients as monolithic.

    Without this the byte counts would be measuring a model that trains
    differently from the one we claim to be training.
    """
    model.zero_grad(set_to_none=True)
    _, loss_mono = model(idx, targets)
    loss_mono.backward()
    g_mono = _flat_grads(model).clone()

    model.zero_grad(set_to_none=True)
    loss_pipe, _ = pipeline_step(model, idx, targets, n_stages, compress=False)
    g_pipe = _flat_grads(model).clone()

    dl = abs(loss_mono.item() - loss_pipe.item())
    dg = (g_mono - g_pipe).abs().max().item()
    return {"loss_delta": dl, "max_grad_delta": dg,
            "equivalent": dl < tol and dg < tol}


def training_memory_bytes(n_params: int) -> int:
    """bf16 weights + fp32 master + Adam m,v  =  2 + 4 + 8 bytes per parameter."""
    return n_params * 14
