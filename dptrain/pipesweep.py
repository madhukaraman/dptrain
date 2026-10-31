"""Does pipeline parallelism let small devices join a big run?

Two parts. First a measurement: sweep d_model and check whether arithmetic
intensity at a stage boundary really grows linearly with it, which is the
claim that would make pipelining over slow links viable at scale. Then a
projection onto real model sizes and consumer hardware, where the binding
constraint turns out not to be bandwidth at all.
"""

import argparse
import json
import os

import torch

from .data import VOCAB_SIZE, build_shards, get_batch
from .model import Config, TinyGPT
from .netsim import PRESETS, Link
from .pipeline import (PipelineStats, pipeline_step, training_memory_bytes,
                       verify_equivalence)

# (name, params, d_model, n_layer)
REAL_MODELS = [("7B", 7e9, 4096, 32), ("13B", 13e9, 5120, 40), ("70B", 70e9, 8192, 80)]


def measure_intensity(d_models, n_layer=8, n_stages=4, batch=8, block=64, seed=0):
    """FLOPs per boundary byte, measured not assumed."""
    rows = []
    for d in d_models:
        cfg = Config(vocab_size=VOCAB_SIZE, n_layer=n_layer, d_model=d, block_size=block)
        torch.manual_seed(seed)
        m = TinyGPT(cfg)
        shards, _ = build_shards(1, 20000, non_iid=False, seed=seed)
        rng = torch.Generator().manual_seed(seed)
        x, y = get_batch(shards[0], batch, block, rng, "cpu")

        m.zero_grad(set_to_none=True)
        st = PipelineStats()
        pipeline_step(m, x, y, n_stages, compress=False, stats=st)

        tokens = batch * block
        flops_stage = 6 * st.peak_stage_params * tokens
        per_boundary = st.bytes_per_microbatch // max(1, n_stages - 1)
        rows.append({
            "d_model": d,
            "stage_params": st.peak_stage_params,
            "bytes_per_boundary": per_boundary,
            "flops_per_stage": flops_stage,
            "intensity": round(flops_stage / per_boundary, 1),
        })
    return rows


def project(params, d_model, n_layer, n_stages, device, link,
            micro_batch=1, seq=2048, act_bytes=1):
    """One (model, stage-count) configuration on real hardware.

    act_bytes=1 is int8 activations. Unlike int8 pseudo-gradients this is
    genuinely lossy -- an activation is consumed once, so there is no
    error-feedback buffer to carry the residual forward.
    """
    per_stage_params = params / n_stages
    mem = training_memory_bytes(per_stage_params)
    tokens = micro_batch * seq

    compute_s = device.compute_seconds(tokens, 6 * per_stage_params)
    boundary_bytes = 2 * micro_batch * seq * d_model * act_bytes  # fwd + bwd
    comm_s = link.transfer_seconds(boundary_bytes)

    return {
        "n_stages": n_stages,
        "params_per_stage": int(per_stage_params),
        "layers_per_stage": round(n_layer / n_stages, 2),
        "memory_gb": round(mem / 1e9, 1),
        "fits_24gb": mem <= 24e9,
        "compute_s": round(compute_s, 4),
        "comm_s": round(comm_s, 4),
        "boundary_mb": round(boundary_bytes / 1e6, 2),
        # No overlap: strictly sequential send-then-compute.
        "util_no_overlap": round(compute_s / (compute_s + comm_s), 4),
        # Steady state with enough micro-batches in flight to hide the send.
        "util_overlapped": round(min(1.0, compute_s / max(compute_s, comm_s)), 4),
        "intensity": round(6 * per_stage_params * tokens / boundary_bytes, 1),
    }


def min_stages_for(params, budget_bytes):
    """Fewest stages whose per-stage training state fits the budget."""
    import math
    return math.ceil(training_memory_bytes(params) / budget_bytes)


def island_availability(n_stages, per_device=0.95, redundancy=1):
    """A pipeline is only as available as its least reliable stage."""
    stage_up = 1 - (1 - per_device) ** redundancy
    return stage_up ** n_stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-preset", default="rtx4090", choices=list(PRESETS))
    ap.add_argument("--uplink-mbps", type=float, default=110.0)
    ap.add_argument("--rtt-ms", type=float, default=80.0)
    ap.add_argument("--vram-gb", type=float, default=24.0)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev, link = PRESETS[args.device_preset], Link(args.uplink_mbps, args.rtt_ms)
    budget = args.vram_gb * 1e9

    print("CORRECTNESS")
    cfg = Config(vocab_size=VOCAB_SIZE, n_layer=8, d_model=128, block_size=64)
    torch.manual_seed(0)
    m = TinyGPT(cfg)
    shards, _ = build_shards(1, 20000, non_iid=False, seed=0)
    x, y = get_batch(shards[0], 8, 64, torch.Generator().manual_seed(0), "cpu")
    checks = {ns: verify_equivalence(m, x, y, ns) for ns in (2, 4, 8)}
    for ns, r in checks.items():
        print(f"  {ns} stages: grads match monolithic "
              f"(max delta {r['max_grad_delta']:.1e})" if r["equivalent"]
              else f"  {ns} stages: MISMATCH {r}")

    print("\nMEASURED — arithmetic intensity vs d_model (4 stages, 8 layers)")
    rows = measure_intensity([64, 128, 256, 512])
    print(f"  {'d_model':>8}{'bytes/bdry':>12}{'FLOP/stage':>14}{'FLOP/byte':>11}{'ratio':>8}")
    base = None
    for r in rows:
        base = base or r["intensity"]
        print(f"  {r['d_model']:>8}{r['bytes_per_boundary']:>12,}"
              f"{r['flops_per_stage']:>14,}{r['intensity']:>11.1f}"
              f"{r['intensity']/base:>8.2f}x")
    print("  (if intensity is linear in d_model, ratio should track d_model/64)")

    print(f"\nPROJECTED — {dev.name}, {args.uplink_mbps:.0f} Mb/s up, "
          f"{args.vram_gb:.0f}GB VRAM, int8 activations, "
          f"micro-batch {args.micro_batch}x{args.seq}")
    report = {"measured_intensity": rows, "projections": {}}
    for name, params, d_model, n_layer in REAL_MODELS:
        need = min_stages_for(params, budget)
        print(f"\n  {name}  ({d_model} d_model, {n_layer} layers)")
        print(f"    minimum stages to fit {args.vram_gb:.0f}GB: {need}"
              f"   ({n_layer/need:.1f} layers per card)" if need <= n_layer
              else f"    minimum stages to fit {args.vram_gb:.0f}GB: {need}"
                   f"   IMPOSSIBLE — more stages than layers ({n_layer})")
        cfgs = []
        for ns in sorted({need, need * 2, max(need, n_layer)}):
            if ns > n_layer:
                continue
            p = project(params, d_model, n_layer, ns, dev, link,
                        args.micro_batch, args.seq)
            p["island_availability"] = round(island_availability(ns), 4)
            p["island_availability_2x"] = round(island_availability(ns, redundancy=2), 4)
            cfgs.append(p)
            print(f"    {ns:>3} stages  mem/card {p['memory_gb']:>6.1f}GB  "
                  f"{'fits' if p['fits_24gb'] else 'OVER'}  "
                  f"util {p['util_overlapped']*100:>5.1f}% (overlapped) "
                  f"{p['util_no_overlap']*100:>5.1f}% (serial)  "
                  f"island up {p['island_availability']*100:>5.1f}% "
                  f"/ {p['island_availability_2x']*100:>5.1f}% at 2x")
        report["projections"][name] = cfgs

    out = args.out or os.path.join(os.environ.get("BB_THREAD_STORAGE", "/tmp"),
                                   "pipeline_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
