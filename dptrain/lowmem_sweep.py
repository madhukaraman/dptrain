"""Measure what low-memory optimizer states actually cost in quality.

Single-worker training, so the only variable is the optimizer. DiLoCo adds its
own noise; isolating this first makes the signal readable.
"""

import argparse
import json
import os
import time

import torch

from .data import VOCAB_SIZE, build_shards, get_batch
from .diloco import evaluate
from .lowmem import CONFIGS, STATE_BYTES_PER_PARAM, LowMemAdamW
from .model import Config, TinyGPT


def train(name, cfg_kwargs, cfg, data, held_out, steps, batch, lr, seed, device):
    torch.manual_seed(seed)
    model = TinyGPT(cfg).to(device)
    opt = LowMemAdamW(model.parameters(), lr=lr, **cfg_kwargs)
    rng = torch.Generator().manual_seed(seed)
    curve, t0 = [], time.time()
    for s in range(steps):
        x, y = get_batch(data, batch, cfg.block_size, rng, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (s + 1) % max(1, steps // 20) == 0:
            curve.append(round(evaluate(model, held_out, batch, cfg.block_size,
                                        device, iters=10), 4))
    return {"name": name, "curve": curve, "final": curve[-1],
            "seconds": round(time.time() - t0, 1),
            "state_bytes_per_param": STATE_BYTES_PER_PARAM[name]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config(vocab_size=VOCAB_SIZE, n_layer=args.n_layer,
                 d_model=args.d_model, block_size=64)
    shards, held_out = build_shards(1, 120000, non_iid=False, seed=args.seed)
    data = shards[0]

    probe = TinyGPT(cfg)
    print(f"model {probe.n_params()/1e6:.2f}M params | {args.steps} steps | "
          f"batch {args.batch_size}\n")

    results = []
    for name, kw in CONFIGS.items():
        r = train(name, kw, cfg, data, held_out, args.steps, args.batch_size,
                  args.lr, args.seed, args.device)
        results.append(r)
        print(f"  {name:<28} final {r['final']:.4f}  "
              f"({r['state_bytes_per_param']:>2} B/param state)  {r['seconds']:>5.1f}s")

    base = results[0]["final"]
    print(f"\n  {'config':<28}{'final':>9}{'vs fp32':>10}{'total B/param':>15}")
    out = []
    for r in results:
        delta = (r["final"] - base) / base * 100
        total = 2 + 2 + r["state_bytes_per_param"]  # bf16 weights + bf16 grads
        print(f"  {r['name']:<28}{r['final']:>9.4f}{delta:>9.2f}%{total:>15}")
        out.append({**r, "pct_vs_fp32": round(delta, 3), "total_bytes_per_param": total})

    path = args.out or os.path.join(os.environ.get("BB_THREAD_STORAGE", "/tmp"),
                                    "lowmem_report.json")
    json.dump({"config": vars(args), "results": out}, open(path, "w"), indent=2)
    print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
