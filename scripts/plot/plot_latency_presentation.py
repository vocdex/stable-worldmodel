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
import matplotlib.patches as mpatches
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
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Feature space per frame", fontsize=22, pad=14)

    # --- DINO: 16×16 grid of tiny squares ---
    grid = 16
    sq = 0.22
    gap = 0.005
    origin_x, origin_y = 0.3, 1.6
    for r in range(grid):
        for c in range(grid):
            x = origin_x + c * (sq + gap)
            y = origin_y + r * (sq + gap)
            rect = mpatches.FancyBboxPatch(
                (x, y), sq, sq,
                boxstyle="square,pad=0",
                facecolor=DINO_FILL, edgecolor=DINO_EDGE, linewidth=0.3,
            )
            ax.add_patch(rect)

    dino_top = origin_y + grid * (sq + gap) + 0.05
    dino_cx  = origin_x + grid * (sq + gap) / 2
    ax.text(dino_cx, dino_top + 0.05, "256 tokens", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=DINO_EDGE)
    ax.text(dino_cx, dino_top + 0.52, "384-d each", ha="center", va="bottom",
            fontsize=12, color="#444")
    ax.text(dino_cx, origin_y - 0.55, "98,304 values", ha="center", va="top",
            fontsize=11, color="#555", style="italic")

    # --- Arrow ---
    arrow_x = origin_x + grid * (sq + gap) + 0.35
    arrow_cx = arrow_x + 0.55
    mid_y = origin_y + grid * (sq + gap) / 2
    ax.annotate("", xy=(arrow_x + 1.1, mid_y), xytext=(arrow_x, mid_y),
                arrowprops=dict(arrowstyle="-|>", color="#555",
                                lw=2.2, mutation_scale=18))
    ax.text(arrow_cx, mid_y + 0.38, "192×", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#555")
    ax.text(arrow_cx, mid_y - 0.38, "fewer", ha="center", va="top",
            fontsize=11, color="#777")

    # --- Slots: 4 larger squares in a column ---
    n_slots = 4
    sl_w, sl_h = 0.78, 0.72
    sl_gap = 0.18
    sl_x = arrow_x + 1.25
    total_h = n_slots * sl_h + (n_slots - 1) * sl_gap
    sl_y0 = mid_y - total_h / 2

    for i in range(n_slots):
        y = sl_y0 + i * (sl_h + sl_gap)
        rect = mpatches.FancyBboxPatch(
            (sl_x, y), sl_w, sl_h,
            boxstyle="round,pad=0.04",
            facecolor=SC_FILL, edgecolor=SC_EDGE, linewidth=1.6,
        )
        ax.add_patch(rect)

    sl_cx = sl_x + sl_w / 2
    sl_top = sl_y0 + total_h
    ax.text(sl_cx, sl_top + 0.1, "4 tokens", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=SC_EDGE)
    ax.text(sl_cx, sl_top + 0.57, "128-d each", ha="center", va="bottom",
            fontsize=12, color="#444")
    ax.text(sl_cx, sl_y0 - 0.55, "512 values", ha="center", va="top",
            fontsize=11, color="#555", style="italic")


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
