"""Presentation figure: feature-space compression (left) + planning latency (right).

Token counts:
  DINO ViT-S/16 at 256px → 16×16 = 256 patch tokens, 384-d each = 98,304 values
  SlotContrast 7-slot encoder → 4 meaningful slots, 128-d each = 512 values

Planning latency (RTX 4080, fp32, B=1, N=300, 30 CEM iters, H=5, averaged PushT+Cube):
  SlotContrast-WM  5.4 s
  DINO-WM         108 s
  LeWM              2.3 s

    python scripts/plot/plot_latency_presentation.py \
        --out visualizations/latency_presentation.png
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np

DINO_FILL  = "#a9c5e0"; DINO_EDGE  = "#5b8fc4"
SC_FILL    = "#f0c97a"; SC_EDGE    = "#d99a2b"
LEWM_FILL  = "#cacaca"; LEWM_EDGE  = "#8f8f8f"

LATENCY = [
    ("SlotContrast-WM\n(ours)", 5.4,  SC_FILL,   SC_EDGE),
    ("DINO-WM",                108.0, DINO_FILL, DINO_EDGE),
    ("LeWM",                   2.3,   LEWM_FILL, LEWM_EDGE),
]


def plot_token_reduction(ax):
    ax.axis("off")
    ax.set_title("Feature space per frame", fontsize=22, pad=14)

    # DINO-WM row
    ax.text(0.5, 0.72, "DINO-WM", transform=ax.transAxes,
            fontsize=17, fontweight="bold", color=DINO_EDGE, ha="center")
    ax.text(0.5, 0.55, "256 × 384-d", transform=ax.transAxes,
            fontsize=26, color="#222", ha="center", fontfamily="monospace")

    # Arrow
    ax.text(0.5, 0.38, "↓  99.5% fewer values", transform=ax.transAxes,
            fontsize=17, color="#888", ha="center", style="italic")

    # SlotContrast-WM row
    ax.text(0.5, 0.24, "SlotContrast-WM (ours)", transform=ax.transAxes,
            fontsize=17, fontweight="bold", color=SC_EDGE, ha="center")
    ax.text(0.5, 0.07, "4 × 128-d", transform=ax.transAxes,
            fontsize=26, color="#222", ha="center", fontfamily="monospace")


def plot_latency(ax):
    labels = [d[0] for d in LATENCY]
    vals   = [d[1] for d in LATENCY]
    fills  = [d[2] for d in LATENCY]
    edges  = [d[3] for d in LATENCY]

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.62, color=fills, edgecolor=edges, linewidth=2.5)
    ax.set_yscale("log")
    ax.set_ylim(min(vals) * 0.45, max(vals) * 3)

    def fmt(v):
        return f"{v:.0f} s" if v >= 10 else f"{v:.1f} s"

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.16, fmt(v),
                ha="center", va="bottom", fontsize=20, color="#222222")

    ax.set_title("Planning time per replan", fontsize=22, pad=14)
    ax.set_ylabel("Time (s)", fontsize=19)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=14)
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="visualizations/latency_presentation.png")
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans"})

    fig, (ax_tok, ax_lat) = plt.subplots(1, 2, figsize=(14, 6.5))
    plot_token_reduction(ax_tok)
    plot_latency(ax_lat)
    fig.tight_layout(w_pad=3.5)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
