"""Run a DiLoCo simulation against a centralized baseline and emit a JSON report."""

import argparse
import json
import os
import time

import torch

from .data import VOCAB_SIZE, build_shards, get_batch
from .diloco import OuterOptimizer, clone_state, evaluate, make_peer
from .model import Config, TinyGPT
from .netsim import PRESETS, Churn, Link, project, project_sweep


def _baseline_once(cfg, pooled, held_out, args, device, lr, batch):
    torch.manual_seed(args.seed)
    model = TinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    rng = torch.Generator().manual_seed(args.seed)
    curve = []
    for step in range(args.rounds * args.inner_steps):
        x, y = get_batch(pooled, batch, cfg.block_size, rng, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.inner_steps == 0:
            curve.append(evaluate(model, held_out, args.batch_size,
                                  cfg.block_size, device))
    return curve


def run_baseline(cfg, shards, held_out, args, device):
    """Fully synchronous data-parallel equivalent: what the same peers would
    achieve over a perfect network.

    This matches DiLoCo on both total tokens and optimizer steps -- effective
    batch is batch_size x n_peers over rounds x inner_steps steps. The earlier
    version used a single worker at batch_size, which handed the baseline 1/K
    of the compute and made DiLoCo look better than it is.

    We try the unscaled and sqrt(K)-scaled learning rate and keep the better
    curve, so the comparison is against a tuned baseline rather than a
    handicapped one.
    """
    pooled = torch.cat(shards)
    eff_batch = args.batch_size * args.peers
    candidates = [args.inner_lr, args.inner_lr * (args.peers ** 0.5)]
    best, best_lr = None, None
    for lr in candidates:
        curve = _baseline_once(cfg, pooled, held_out, args, device, lr, eff_batch)
        if best is None or curve[-1] < best[-1]:
            best, best_lr = curve, lr
    print(f"  baseline: batch {eff_batch}, {args.rounds * args.inner_steps} steps, "
          f"best lr {best_lr:.2e} -> {best[-1]:.4f}")
    return best


def run_diloco(cfg, shards, held_out, args, device):
    torch.manual_seed(args.seed)
    outer = OuterOptimizer(clone_state(TinyGPT(cfg).to(device)),
                           lr=args.outer_lr, momentum=0.9)
    peers = [make_peer(k, shards[k], cfg, args.inner_lr, device)
             for k in range(args.peers)]
    churn = Churn(drop_prob=args.drop_prob, leave_prob=args.leave_prob,
                  join_prob=args.join_prob, max_peers=args.peers + 4)
    link = Link(uplink_mbps=args.uplink_mbps, rtt_ms=args.rtt_ms)
    crng = torch.Generator().manual_seed(args.seed + 7)
    drng = torch.Generator().manual_seed(args.seed + 11)

    probe = TinyGPT(cfg).to(device)
    tokens_per_peer_round = args.inner_steps * args.batch_size * cfg.block_size

    curve, rounds_log, all_events = [], [], []
    bytes_total = 0
    wall_start = time.time()

    for r in range(args.rounds):
        contributing, events, joined = churn.apply(peers, crng)
        if joined:
            new = make_peer(len(peers), shards[len(peers) % len(shards)], cfg,
                            args.inner_lr, device, round_idx=r)
            peers.append(new)
            contributing.append(new)
            events.append(("join", new.peer_id))

        grads, round_bytes, losses = [], 0, []
        for p in contributing:
            losses.append(p.inner_steps(outer.state, args.inner_steps,
                                        args.batch_size, cfg.block_size,
                                        drng, device))
            g, nbytes = p.pseudo_gradient(outer.state, compress=not args.no_compress)
            grads.append(g)
            round_bytes += nbytes

        outer.step(grads)
        bytes_total += round_bytes

        probe.load_state_dict(outer.state)
        val = evaluate(probe, held_out, args.batch_size, cfg.block_size, device)
        curve.append(val)
        rounds_log.append({
            "round": r,
            "contributing": len(contributing),
            "live": sum(1 for p in peers if p.alive),
            "bytes": round_bytes,
            "train_loss": round(sum(losses) / max(1, len(losses)), 4),
            "val_loss": round(val, 4),
            "events": events,
        })
        all_events += [{"round": r, "kind": k, "peer": pid} for k, pid in events]
        if args.verbose:
            print(f"  round {r:>3}  peers {len(contributing):>2}  "
                  f"val {val:.4f}  {round_bytes/1024:.0f} KiB"
                  + (f"  {events}" if events else ""))

    wall = time.time() - wall_start
    per_peer_bytes = bytes_total / max(1, sum(r["contributing"] for r in rounds_log))
    projections = [
        project(int(per_peer_bytes), probe.n_params(), int(target),
                tokens_per_peer_round, args.inner_steps,
                PRESETS[args.device_preset], link)
        for target in args.project_params
    ]
    sweep = project_sweep(int(per_peer_bytes), probe.n_params(),
                          int(args.project_params[-1]),
                          args.project_tokens_per_step,
                          [10, 30, 50, 100, 250, 500, 1000, 2000],
                          PRESETS[args.device_preset], link)
    return {
        "curve": curve,
        "rounds": rounds_log,
        "events": all_events,
        "bytes_total": bytes_total,
        "bytes_per_peer_round": int(per_peer_bytes),
        "wall_seconds": round(wall, 1),
        "projections": projections,
        "sweep": sweep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--inner-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--inner-lr", type=float, default=3e-4)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--chars-per-peer", type=int, default=60000)
    ap.add_argument("--iid", action="store_true", help="uniform shards instead of non-IID")
    ap.add_argument("--no-compress", action="store_true", help="send fp32 pseudo-gradients")
    ap.add_argument("--drop-prob", type=float, default=0.05)
    ap.add_argument("--leave-prob", type=float, default=0.02)
    ap.add_argument("--join-prob", type=float, default=0.03)
    ap.add_argument("--uplink-mbps", type=float, default=110.0)
    ap.add_argument("--rtt-ms", type=float, default=80.0)
    ap.add_argument("--device-preset", default="rtx4090", choices=list(PRESETS))
    ap.add_argument("--project-params", type=float, nargs="+",
                    default=[1e9, 7e9, 70e9],
                    help="model sizes to project utilization for")
    ap.add_argument("--project-tokens-per-step", type=int, default=192 * 2048,
                    help="tokens per inner step on a real peer (batch x seq)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config(vocab_size=VOCAB_SIZE, n_layer=args.n_layer,
                 d_model=args.d_model, block_size=args.block_size)
    shards, held_out = build_shards(args.peers, args.chars_per_peer,
                                    non_iid=not args.iid, seed=args.seed)

    probe = TinyGPT(cfg)
    print(f"model: {probe.n_params()/1e6:.2f}M params, "
          f"{probe.flops_per_token()/1e6:.1f} MFLOP/token")
    print(f"shards: {args.peers} peers x {len(shards[0])} tokens, "
          f"{'IID' if args.iid else 'non-IID'}")

    print("running DiLoCo...")
    diloco = run_diloco(cfg, shards, held_out, args, args.device)

    baseline = None
    if not args.skip_baseline:
        print("running centralized baseline...")
        baseline = run_baseline(cfg, shards, held_out, args, args.device)

    report = {
        "config": vars(args),
        "model": {"params": probe.n_params(),
                  "flops_per_token": probe.flops_per_token()},
        "diloco": diloco,
        "baseline_curve": baseline,
    }
    out = args.out or os.path.join(
        os.environ.get("BB_THREAD_STORAGE", "/tmp"), "diloco_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nfinal val loss  diloco={diloco['curve'][-1]:.4f}"
          + (f"  baseline={baseline[-1]:.4f}" if baseline else ""))
    print(f"bytes/peer/round: {diloco['bytes_per_peer_round']/1024:.1f} KiB")
    sw = diloco["sweep"]
    print(f"\nutilization vs inner steps  [{sw['target_params']/1e9:.0f}B on "
          f"{sw['device']}, {sw['uplink_mbps']:.0f} Mb/s up, "
          f"{sw['round_bytes']/1e9:.2f} GB/round, {sw['comm_s']:.0f}s to send]")
    for pt in sw["points"]:
        bar = "#" * int(pt["utilization"] * 40)
        print(f"  H={pt['inner_steps']:>5}  {pt['utilization']*100:5.1f}%  {bar}")
    print(f"  -> H >= {sw['exact_h_for_90pct']:.0f} for 90% utilization")

    print(f"\nprojected on {diloco['projections'][0]['device']} "
          f"@ {args.uplink_mbps:.0f} Mb/s up, {args.inner_steps} inner steps:")
    for p in diloco["projections"]:
        print(f"  {p['target_params']/1e9:>5.0f}B params -> "
              f"util {p['diloco_utilization']*100:5.1f}%  "
              f"(naive DDP {p['ddp_utilization']*100:.4f}%)  "
              f"{p['comms_reduction_x']}x less traffic")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
