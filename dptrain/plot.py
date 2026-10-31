"""Render the benchmark charts for the README.

Two variants per chart -- light and dark -- so GitHub can swap them with
<picture media="prefers-color-scheme">. The dark steps are chosen for the dark
surface rather than being an automatic inversion of the light ones.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e",
        "grid": "#e5e4e0", "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7",
        "grid": "#333331", "series": ["#3987e5", "#d95926", "#199e70"],
    },
}


def _style(ax, th, xlabel, ylabel):
    ax.set_facecolor(th["surface"])
    ax.figure.set_facecolor(th["surface"])
    ax.grid(True, color=th["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["grid"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=th["text2"], labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=th["text2"], fontsize=10)
    ax.set_ylabel(ylabel, color=th["text2"], fontsize=10)


def _label_end(ax, x, y, text, color, dy=0.0):
    """Direct label at the series end -- also the relief required by the
    light-mode contrast warning on the aqua slot."""
    ax.annotate(text, xy=(x, y), xytext=(6, dy), textcoords="offset points",
                color=color, fontsize=9, fontweight="bold",
                va="center", ha="left")


def convergence(summary, theme, out):
    th = THEMES[theme]
    r = summary["runs"]
    rounds = list(range(1, len(r["main"]["curve"]) + 1))
    series = [
        ("Synchronous DDP", r["main"]["baseline_curve"], th["series"][0], 0),
        ("DiLoCo, IID", r["iid"]["curve"], th["series"][1], -9),
        ("DiLoCo, non-IID + churn", r["main"]["curve"], th["series"][2], 9),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=200)
    for name, ys, color, dy in series:
        ax.plot(rounds, ys, color=color, linewidth=2.0, label=name,
                solid_capstyle="round")
        _label_end(ax, rounds[-1], ys[-1], f"{ys[-1]:.3f}", color, dy)
    _style(ax, th, "Outer round", "Held-out loss (log)")
    # Linear y buries the result: all three curves flatten and the 10.3% gap
    # we care about is a pixel wide. Log y keeps the separation legible.
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1.0, 2.0, 3.0])
    ax.set_yticklabels(["0.25", "0.5", "1.0", "2.0", "3.0"])
    ax.minorticks_off()
    ax.set_xlim(1, len(rounds) + 3.6)
    ax.set_ylim(0.21, 3.3)
    ax.set_title("Convergence under non-IID shards and peer churn",
                 color=th["text"], fontsize=12, fontweight="bold",
                 loc="left", pad=12)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=9, ncol=1)
    for t in leg.get_texts():
        t.set_color(th["text2"])
    fig.tight_layout()
    fig.savefig(out, facecolor=th["surface"])
    plt.close(fig)


def utilization(summary, theme, out):
    th = THEMES[theme]
    pts = summary["sweep"]["points"]
    hs = [p["inner_steps"] for p in pts]
    int8 = [p["utilization"] * 100 for p in pts]
    # fp32 payload is 4x larger; recompute from the same compute/comm model.
    fp32 = []
    for p in pts:
        comp, comm = p["compute_s"], p["comm_s"] * 4
        fp32.append(comp / (comp + comm) * 100)

    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=200)
    for name, ys, color, dy in [("int8 + error feedback", int8, th["series"][0], 7),
                                ("fp32 pseudo-gradients", fp32, th["series"][1], -7)]:
        ax.plot(hs, ys, color=color, linewidth=2.0, marker="o", markersize=4.5,
                label=name, solid_capstyle="round")
        _label_end(ax, hs[-1], ys[-1], f"{ys[-1]:.1f}%", color, dy)
    ax.axhline(90, color=th["text2"], linewidth=1.0, linestyle=(0, (4, 4)),
               alpha=0.6)
    ax.annotate("90% utilization", xy=(hs[0], 90), xytext=(0, 6),
                textcoords="offset points", color=th["text2"], fontsize=8.5)
    for x, color, label in [(16, th["series"][0], "H=16"),
                            (64, th["series"][1], "H=64")]:
        ax.plot([x, x], [40, 90], color=color, linewidth=1.2,
                linestyle=(0, (3, 3)), alpha=0.75)
        ax.annotate(label, xy=(x, 40), xytext=(0, 5), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold", ha="center")
    ax.set_xscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs])
    _style(ax, th, "Inner steps between syncs (H)", "Projected utilization (%)")
    ax.set_xlim(hs[0] * 0.85, hs[-1] * 1.9)
    ax.set_ylim(40, 104)
    ax.set_title("Utilization vs sync interval — 70B on an RTX 4090, 110 Mb/s up",
                 color=th["text"], fontsize=12, fontweight="bold",
                 loc="left", pad=12)
    leg = ax.legend(frameon=False, loc="lower right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(th["text2"])
    fig.tight_layout()
    fig.savefig(out, facecolor=th["surface"])
    plt.close(fig)


def main(summary_path, out_dir):
    with open(summary_path) as f:
        summary = json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    for theme in ("light", "dark"):
        convergence(summary, theme, os.path.join(out_dir, f"convergence-{theme}.png"))
        utilization(summary, theme, os.path.join(out_dir, f"utilization-{theme}.png"))
    print(f"wrote 4 charts to {out_dir}")


if __name__ == "__main__":
    store = os.environ.get("BB_THREAD_STORAGE", "/tmp")
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(store, "summary.json"),
         sys.argv[2] if len(sys.argv) > 2 else "docs")
