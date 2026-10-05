# -*- coding: utf-8 -*-
"""
Revised data-region figures — round 2 polish v2.
Figure 2 label offsets enlarged so they sit clearly OUTSIDE the markers.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    path = ROOT / "Times new Roman" / name
    if path.exists():
        font_manager.fontManager.addfont(str(path))

plt.rcParams.update({"font.family": "Times New Roman", "axes.linewidth": 0.9,
                     "savefig.dpi": 300})

ACCENT      = "#0e7c66"
DARK_SLATE  = "#475569"
MID_SLATE   = "#94a3b8"
LIGHT_SLATE = "#cbd5e1"
REF_RED     = "#dc2626"
INK         = "#1f2937"
MUTE        = "#6b7280"

DATASETS = ["AdvBench", "HarmBench", "JailbreakBench", "Overall"]
DATA = {
    "LLaMA3-8B": {"Merged": [50.0, 54.0, 62.0, 55.3],
                  "Chat":   [28.0, 36.0, 46.0, 36.7],
                  "Base":   [28.0, 32.0, 36.0, 32.0]},
    "LLaMA3-3B": {"Merged": [64.0, 68.0, 66.0, 66.0],
                  "Chat":   [14.0, 16.0, 24.0, 18.0],
                  "Base":   [42.0, 30.0, 46.0, 39.3]},
    "SmolLM2 (1.7B)": {"Merged": [48.0, 54.0, 40.0, 47.3],
                       "Chat":   [26.0, 20.0, 10.0, 18.7],
                       "Base":   [22.0, 22.0, 28.0, 24.0]},
}


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / ("%s.%s" % (stem, ext)), bbox_inches="tight",
                    pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print("saved", stem)


def fig_search_compare():
    metrics = [
        ("Capability retention (%)", "higher is better", 73.4, 80.0, None),
        ("Merged-model ASR (%)",     "higher is better", 52.7, 55.3, 36.7),
        ("Base weight share (%)",    "lower is better",  49.2, 26.3, None),
    ]
    OFF = 4.5         # label offset (data units) - sit outside markers
    MARKER_S = 160
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    n = len(metrics)
    ys = list(range(n - 1, -1, -1))
    for y, (label, hint, g, p, ref) in zip(ys, metrics):
        lo, hi = (g, p) if g <= p else (p, g)
        ax.plot([lo, hi], [y, y], color=ACCENT, lw=5.0, alpha=0.28,
                solid_capstyle="round", zorder=2)
        ax.scatter([g], [y], s=MARKER_S, color=MID_SLATE, edgecolor="white",
                   lw=1.6, zorder=5)
        ax.scatter([p], [y], s=MARKER_S, color=ACCENT, edgecolor="white",
                   lw=1.6, zorder=5)
        if g <= p:
            ax.text(g - OFF, y, "%.1f" % g, ha="right", va="center",
                    fontsize=9.5, color=MID_SLATE, fontweight="bold")
            ax.text(p + OFF, y, "%.1f" % p, ha="left", va="center",
                    fontsize=9.5, color=ACCENT, fontweight="bold")
        else:
            ax.text(g + OFF, y, "%.1f" % g, ha="left", va="center",
                    fontsize=9.5, color=MID_SLATE, fontweight="bold")
            ax.text(p - OFF, y, "%.1f" % p, ha="right", va="center",
                    fontsize=9.5, color=ACCENT, fontweight="bold")
        ax.text(123, y, "%s  ·  Δ %+.1f" % (hint, p - g),
                ha="left", va="center", fontsize=8.3,
                color=MUTE, style="italic")
        if ref is not None:
            ax.plot([ref, ref], [y - 0.32, y + 0.32], color=REF_RED,
                    lw=1.6, ls="--", alpha=0.9, zorder=3)
            ax.text(ref, y - 0.48, "Chat baseline %.1f%%" % ref,
                    ha="center", va="top", fontsize=8, color=REF_RED)
    ax.set_yticks(ys)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=9.8, color=INK)
    ax.set_xlim(-8, 122)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100"], fontsize=8.8)
    ax.set_xlabel("Percentage (%)", fontsize=9.8, color=INK)
    ax.set_ylim(-0.85, n - 0.15)
    ax.set_title("LLaMA3-8B: global scalar vs per-layer search "
                 "(no system prompt)", loc="left", weight="bold",
                 fontsize=11.4, color=INK, pad=24)
    ax.grid(axis="x", color="#e5e7eb", lw=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors=MUTE)
    handles = [
        Line2D([], [], marker="o", color="white", markerfacecolor=MID_SLATE,
               markeredgecolor="white", markersize=11, label="Global scalar"),
        Line2D([], [], marker="o", color="white", markerfacecolor=ACCENT,
               markeredgecolor="white", markersize=11, label="Per-layer"),
        Line2D([], [], color=REF_RED, ls="--", lw=1.6,
               label="Chat-model ASR baseline"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.11), ncol=3, fontsize=8.9)
    fig.tight_layout()
    save(fig, "fig_8b_search_compare")


def dataset_asr_figure(model, stem):
    d = DATA[model]
    x = np.arange(len(DATASETS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    palette = [("Merged", ACCENT), ("Chat", DARK_SLATE), ("Base", LIGHT_SLATE)]
    for i, (name, color) in enumerate(palette):
        off = (i - 1) * w
        bars = ax.bar(x + off, d[name], w, label="%s model" % name,
                      color=color, edgecolor="white", lw=0.7)
        for b, v in zip(bars, d[name]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.1,
                    "%.0f" % v if v == int(v) else "%.1f" % v,
                    ha="center", va="bottom", fontsize=7.4,
                    color=INK if color != LIGHT_SLATE else MUTE)
    ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="#f1f5f9", zorder=0)
    ymax = max(max(v) for v in d.values())
    ax.set_ylim(0, ymax + 14)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=9, color=INK)
    ax.set_ylabel("Attack success rate (%)", fontsize=9.6, color=INK)
    ax.set_title("%s: dataset-level attack success rate" % model,
                 loc="left", weight="bold", fontsize=10.9, color=INK, pad=18)
    ax.grid(axis="y", color="#e5e7eb", lw=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors=MUTE)
    ax.legend(frameon=False, fontsize=8.7, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    save(fig, stem)


def main():
    fig_search_compare()
    dataset_asr_figure("LLaMA3-8B", "fig_8b_dataset_asr")
    dataset_asr_figure("LLaMA3-3B", "fig_3b_dataset_asr")
    dataset_asr_figure("SmolLM2 (1.7B)", "fig_m2_dataset_asr")
    print("output dir:", OUT)


if __name__ == "__main__":
    main()
