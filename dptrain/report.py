"""Consolidate ablation reports into one summary for charting."""

import json
import os
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def main(out_dir):
    names = ["main", "iid", "fp32", "nochurn"]
    reports = {}
    for n in names:
        p = os.path.join(out_dir, f"report_{n}.json")
        if os.path.exists(p):
            reports[n] = load(p)

    main_r = reports.get("main")
    summary = {
        "runs": {
            n: {
                "final_val": round(r["diloco"]["curve"][-1], 4),
                "baseline_final_val": round(r["baseline_curve"][-1], 4)
                if r.get("baseline_curve") else None,
                "curve": [round(v, 4) for v in r["diloco"]["curve"]],
                "baseline_curve": [round(v, 4) for v in r["baseline_curve"]]
                if r.get("baseline_curve") else None,
                "bytes_per_peer_round": r["diloco"]["bytes_per_peer_round"],
                "n_events": len(r["diloco"]["events"]),
                "events": r["diloco"]["events"],
                "wall_seconds": r["diloco"]["wall_seconds"],
            }
            for n, r in reports.items()
        }
    }

    if main_r:
        sw = main_r["diloco"]["sweep"]
        summary["sweep"] = sw
        summary["model"] = main_r["model"]
        comp = reports.get("fp32")
        if comp:
            summary["compression_x"] = round(
                comp["diloco"]["bytes_per_peer_round"]
                / main_r["diloco"]["bytes_per_peer_round"], 2)
        summary["peers_over_time"] = [
            {"round": r["round"], "contributing": r["contributing"],
             "live": r["live"]}
            for r in main_r["diloco"]["rounds"]
        ]

    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "peers_over_time"},
                     indent=2)[:3000])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else os.environ.get("BB_THREAD_STORAGE", "/tmp"))
