"""Network and churn simulation, plus the utilization projection.

We do not try to infer real-hardware utilization from a laptop's wall clock --
that would be meaningless. Instead the simulator measures the two quantities
that are hardware-independent (bytes on the wire, convergence) and projects
utilization analytically for a device/link you specify.
"""

from dataclasses import dataclass

import torch


@dataclass
class Churn:
    drop_prob: float = 0.05      # chance a live peer misses a round
    leave_prob: float = 0.02     # chance a peer leaves permanently
    join_prob: float = 0.03      # chance a new peer joins
    max_peers: int = 12

    def apply(self, peers: list, rng: torch.Generator):
        events = []
        for p in peers:
            if not p.alive:
                continue
            if torch.rand(1, generator=rng).item() < self.leave_prob:
                p.alive = False
                events.append(("leave", p.peer_id))
        live = [p for p in peers if p.alive]
        contributing = []
        for p in live:
            if torch.rand(1, generator=rng).item() < self.drop_prob:
                events.append(("drop", p.peer_id))
            else:
                contributing.append(p)
        joined = (len(live) < self.max_peers
                  and torch.rand(1, generator=rng).item() < self.join_prob)
        return contributing, events, joined


@dataclass
class Link:
    uplink_mbps: float = 110.0
    rtt_ms: float = 80.0

    def transfer_seconds(self, nbytes: int) -> float:
        return nbytes * 8 / (self.uplink_mbps * 1e6) + self.rtt_ms / 1000.0


@dataclass
class Device:
    name: str
    tflops: float      # achievable bf16 dense
    mfu: float = 0.40  # fraction actually realized

    def compute_seconds(self, tokens: int, flops_per_token: int) -> float:
        return tokens * flops_per_token / (self.tflops * 1e12 * self.mfu)


PRESETS = {
    "rtx4090": Device("RTX 4090", 165.0, 0.35),
    "a100": Device("A100 80GB", 312.0, 0.45),
    "h100": Device("H100 SXM", 990.0, 0.45),
    "b200": Device("B200", 2250.0, 0.45),
}


def project(measured_bytes: int, measured_params: int, target_params: int,
            tokens_per_round: int, inner_steps: int,
            device: Device, link: Link) -> dict:
    """Project per-round time at `target_params` scale.

    The simulation measures two scale-free quantities -- bytes on the wire per
    parameter, and convergence. We extrapolate the first to a realistic model
    size rather than reporting the toy model's absolute bytes, which say
    nothing about a real run.
    """
    bytes_per_param = measured_bytes / measured_params
    round_bytes = int(bytes_per_param * target_params)
    flops_per_token = 6 * target_params
    compute = device.compute_seconds(tokens_per_round, flops_per_token)
    diloco_comm = link.transfer_seconds(round_bytes)
    ddp_bytes = target_params * 2 * inner_steps
    ddp_comm = link.transfer_seconds(ddp_bytes)
    return {
        "device": device.name,
        "target_params": target_params,
        "uplink_mbps": link.uplink_mbps,
        "bytes_per_param": round(bytes_per_param, 4),
        "compute_s": round(compute, 3),
        "diloco_comm_s": round(diloco_comm, 3),
        "diloco_bytes": round_bytes,
        "diloco_utilization": round(compute / (compute + diloco_comm), 4),
        "ddp_comm_s": round(ddp_comm, 1),
        "ddp_bytes": ddp_bytes,
        "ddp_utilization": round(compute / (compute + ddp_comm), 6),
        "comms_reduction_x": round(ddp_bytes / max(1, round_bytes), 1),
    }


def project_sweep(measured_bytes: int, measured_params: int, target_params: int,
                  tokens_per_inner_step: int, h_values: list,
                  device: Device, link: Link) -> dict:
    """Utilization as a function of inner steps H.

    Traffic per round is independent of H -- one pseudo-gradient either way --
    while compute scales linearly with it. So H is the single knob that buys
    utilization, and the useful output is the H at which the link stops being
    the bottleneck.

    Note utilization is scale-invariant in model size here: compute and payload
    both scale linearly with parameters, so they cancel. That is specific to
    data-parallel DiLoCo; pipeline parallelism behaves differently because
    activations scale with d_model while compute scales with d_model squared.
    """
    bytes_per_param = measured_bytes / measured_params
    round_bytes = int(bytes_per_param * target_params)
    comm = link.transfer_seconds(round_bytes)
    flops_per_token = 6 * target_params

    points, h_for_90 = [], None
    for h in h_values:
        tokens = h * tokens_per_inner_step
        compute = device.compute_seconds(tokens, flops_per_token)
        util = compute / (compute + comm)
        points.append({"inner_steps": h,
                       "tokens_per_round": tokens,
                       "compute_s": round(compute, 2),
                       "comm_s": round(comm, 2),
                       "utilization": round(util, 4)})
        if h_for_90 is None and util >= 0.90:
            h_for_90 = h

    # Closed form: util >= 0.9  <=>  compute >= 9 * comm
    per_step = device.compute_seconds(tokens_per_inner_step, flops_per_token)
    exact_h_90 = 9 * comm / per_step

    return {
        "device": device.name,
        "target_params": target_params,
        "uplink_mbps": link.uplink_mbps,
        "round_bytes": round_bytes,
        "comm_s": round(comm, 2),
        "tokens_per_inner_step": tokens_per_inner_step,
        "points": points,
        "h_for_90pct": h_for_90,
        "exact_h_for_90pct": round(exact_h_90, 1),
    }
