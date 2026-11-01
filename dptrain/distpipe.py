"""Pipeline stages in separate OS processes, talking over real TCP.

Each rank builds the whole model from a shared seed and then owns one
contiguous slice of layers. Weights are identical everywhere, so no weight
transfer is needed -- only activations forward and their gradients back,
which is the traffic the projections have been counting.

Run one rank per machine by pointing --host at rank 0; nothing else changes.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

from .data import VOCAB_SIZE, build_shards, get_batch
from .model import Config, TinyGPT
from .pipeline import partition
from .transport import Wire, connect, serve


def build(cfg, seed):
    torch.manual_seed(seed)
    return TinyGPT(cfg)


def batch_for(cfg, seed, batch_size):
    """Every rank derives the same batch from the same seed, so targets need
    not cross the wire."""
    shards, _ = build_shards(1, 40000, non_iid=False, seed=seed)
    rng = torch.Generator().manual_seed(seed)
    return get_batch(shards[0], batch_size, cfg.block_size, rng, "cpu")


def worker(rank, n_stages, host, base_port, cfg, seed, batch_size, steps,
           mbps, rtt_ms, out_dir):
    model = build(cfg, seed)
    lo, hi = partition(cfg.n_layer, n_stages)[rank]
    x, y = batch_for(cfg, seed, batch_size)

    # Boundary r sits between rank r and rank r+1; rank r serves it.
    prev_w = next_w = None
    if rank < n_stages - 1:
        next_w = Wire(serve(host, base_port + rank), mbps, rtt_ms)
    if rank > 0:
        prev_w = Wire(connect(host, base_port + rank - 1), mbps, rtt_ms)

    t0 = time.perf_counter()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        if rank == 0:
            h = model.tok(x) + model.pos(torch.arange(cfg.block_size))
        else:
            h = prev_w.recv().requires_grad_(True)
        stage_input = h
        for blk in model.blocks[lo:hi]:
            h = blk(h)

        if rank < n_stages - 1:
            next_w.send(h.detach())
            grad = next_w.recv()
            h.backward(grad)
        else:
            logits = model.head(model.ln_f(h))
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            print(f"[rank {rank}] loss {loss.item():.6f}", flush=True)

        if rank > 0:
            prev_w.send(stage_input.grad)

    wall = time.perf_counter() - t0
    grads = {n: p.grad.clone() for n, p in model.named_parameters()
             if p.grad is not None}
    stats = {"rank": rank, "wall_s": round(wall, 3), "n_grads": len(grads),
             "prev": prev_w.stats() if prev_w else None,
             "next": next_w.stats() if next_w else None}
    torch.save({"grads": grads, "stats": stats},
               os.path.join(out_dir, f"rank{rank}.pt"))
    print(f"[rank {rank}] done in {wall:.2f}s  {json.dumps(stats)}", flush=True)


def monolithic_grads(cfg, seed, batch_size):
    model = build(cfg, seed)
    x, y = batch_for(cfg, seed, batch_size)
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters()
                         if p.grad is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=None, help="internal")
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=59400)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--mbps", type=float, default=0.0,
                    help="throttle the wire; 0 = unthrottled loopback")
    ap.add_argument("--rtt-ms", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = Config(vocab_size=VOCAB_SIZE, n_layer=args.n_layer,
                 d_model=args.d_model, block_size=args.block_size)
    out_dir = args.out_dir or os.path.join(
        os.environ.get("BB_THREAD_STORAGE", "/tmp"), "distpipe")
    os.makedirs(out_dir, exist_ok=True)

    if args.rank is not None:
        worker(args.rank, args.stages, args.host, args.port, cfg, args.seed,
               args.batch_size, args.steps, args.mbps, args.rtt_ms, out_dir)
        return

    print(f"launching {args.stages} ranks on {args.host}:{args.port}+ "
          f"({'unthrottled' if args.mbps == 0 else f'{args.mbps:.0f} Mb/s'})")
    procs = []
    for r in range(args.stages):
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "dptrain.distpipe", "--rank", str(r)]
            + [a for a in sys.argv[1:] if not a.startswith("--rank")]
            + ["--out-dir", out_dir]))
    codes = [p.wait() for p in procs]
    if any(codes):
        print(f"FAILED, exit codes {codes}")
        sys.exit(1)

    loss_mono, g_mono = monolithic_grads(cfg, args.seed, args.batch_size)
    g_dist, stats, wire_bytes = {}, [], 0
    for r in range(args.stages):
        d = torch.load(os.path.join(out_dir, f"rank{r}.pt"), weights_only=False)
        g_dist.update(d["grads"])
        stats.append(d["stats"])
        for side in ("prev", "next"):
            if d["stats"][side]:
                wire_bytes += d["stats"][side]["bytes_sent"]

    missing = set(g_mono) - set(g_dist)
    worst = max((g_mono[n] - g_dist[n]).abs().max().item() for n in g_dist)
    print(f"\nmonolithic loss {loss_mono:.6f}")
    print(f"params with grads: monolithic {len(g_mono)}, distributed {len(g_dist)}"
          f"{'  MISSING ' + str(sorted(missing)) if missing else ''}")
    print(f"max |grad difference| across every parameter: {worst:.3e}  "
          f"{'MATCH' if worst < 1e-6 and not missing else 'MISMATCH'}")
    print(f"bytes actually sent over TCP: {wire_bytes:,} "
          f"({wire_bytes/args.steps:,.0f} per step)")
    for s in stats:
        print(f"  rank {s['rank']}: {s['wall_s']}s")


if __name__ == "__main__":
    main()
