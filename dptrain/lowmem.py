"""Low-memory optimizer states, implemented so the quality cost can be measured
rather than assumed.

The pipeline result turns on dropping from 16 to ~4 bytes per parameter, which
needs two things the projection simply asserted: 8-bit Adam moments, and
dropping the fp32 master copy. Both are numerical-precision questions, so they
reproduce at toy scale -- no GPU required to find out whether they train.

bitsandbytes is not used here: it needs CUDA, and a from-scratch version makes
the failure modes visible. Where this differs from bitsandbytes it is
pessimistic, so treat quality deltas as an upper bound on the real cost.
"""

import torch

BLOCK = 2048


def quantize_blockwise(t: torch.Tensor, block: int = BLOCK, signed: bool = True):
    """Per-block absmax int8. bitsandbytes uses a non-linear dynamic map; this
    linear version is the pessimistic approximation of it."""
    flat = t.flatten()
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-20)
    lim = 127 if signed else 255
    q = torch.round(blocks / absmax * lim).to(torch.int8 if signed else torch.uint8)
    return q, absmax, t.shape, pad


def dequantize_blockwise(q, absmax, shape, pad, signed: bool = True):
    lim = 127 if signed else 255
    out = q.to(torch.float32) * absmax / lim
    out = out.flatten()
    if pad:
        out = out[:-pad]
    return out.view(shape)


def to_bf16(x: torch.Tensor, stochastic: bool) -> torch.Tensor:
    """fp32 -> bf16. bf16 is the top 16 bits of fp32, so plain truncation is
    round-toward-zero: any update smaller than one ULP of the weight vanishes
    and training silently stalls. Stochastic rounding keeps those updates alive
    in expectation."""
    if not stochastic:
        return x.to(torch.bfloat16)
    i = x.view(torch.int32)
    noise = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32, device=x.device)
    return ((i + noise) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)


class LowMemAdamW(torch.optim.Optimizer):
    """AdamW with optional int8 moments and an optional bf16 master copy.

    bytes/param (weights and grads excluded, they are bf16 either way):
        int8_state=False, master="fp32" -> 12   (fp32 master + fp32 m,v)
        int8_state=True,  master="fp32" ->  6   (fp32 master + int8 m,v)
        int8_state=True,  master="bf16" ->  4   (bf16 master + int8 m,v)
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1, int8_state=False, master="fp32",
                 stochastic_rounding=True):
        if master not in ("fp32", "bf16"):
            raise ValueError(master)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))
        self.int8_state = int8_state
        self.master = master
        self.stochastic_rounding = stochastic_rounding

    def _read(self, st, key):
        if not self.int8_state:
            return st[key]
        q, absmax, shape, pad = st[key]
        v = dequantize_blockwise(q, absmax, shape, pad)
        # v is held in sqrt space: it spans orders of magnitude and linear int8
        # across that range is useless. sqrt compresses it into one.
        return v.square() if key == "v" else v

    def _write(self, st, key, val):
        if not self.int8_state:
            st[key] = val
            return
        src = val.clamp_min(0).sqrt() if key == "v" else val
        st[key] = quantize_blockwise(src)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["master"] = (p.detach().float().clone() if self.master == "fp32"
                                    else p.detach().to(torch.bfloat16).clone())
                    self._write(st, "m", torch.zeros_like(g))
                    self._write(st, "v", torch.zeros_like(g))

                st["step"] += 1
                t = st["step"]
                m = self._read(st, "m").mul_(b1).add_(g, alpha=1 - b1)
                v = self._read(st, "v").mul_(b2).addcmul_(g, g, value=1 - b2)
                self._write(st, "m", m)
                self._write(st, "v", v)

                mh = m / (1 - b1 ** t)
                vh = v / (1 - b2 ** t)
                upd = mh / (vh.sqrt() + group["eps"])

                master = st["master"].float()
                master = master - group["lr"] * (upd + group["weight_decay"] * master)
                if self.master == "bf16":
                    st["master"] = to_bf16(master, self.stochastic_rounding)
                    p.copy_(st["master"].float())
                else:
                    st["master"] = master
                    p.copy_(master)
        return loss


CONFIGS = {
    "fp32-adam-fp32-master": dict(int8_state=False, master="fp32"),
    "int8-adam-fp32-master": dict(int8_state=True, master="fp32"),
    "int8-adam-bf16-nearest": dict(int8_state=True, master="bf16",
                                   stochastic_rounding=False),
    "int8-adam-bf16-stochastic": dict(int8_state=True, master="bf16",
                                      stochastic_rounding=True),
}

STATE_BYTES_PER_PARAM = {
    "fp32-adam-fp32-master": 4 + 8,
    "int8-adam-fp32-master": 4 + 2,
    "int8-adam-bf16-nearest": 2 + 2,
    "int8-adam-bf16-stochastic": 2 + 2,
}
