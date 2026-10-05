"""Figure 5 drawing layout; data is supplied by the entry point."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ACCENT = "#0e7c66"
MID_SLATE = "#94a3b8"
REF_RED = "#dc2626"
INK = "#1f2937"
MUTE = "#6b7280"

def fig_search_compare(global_coefficient, layer_coefficient, global_metrics, layer_metrics, chat_asr, save):
    metrics = [
        ("Capability retention (%)", "higher is better", global_metrics["dqr"], layer_metrics["dqr"], None),
        ("Merged-model ASR (%)",     "higher is better", global_metrics["asr"], layer_metrics["asr"], chat_asr),
        ("Mean SLERP coeff. (%)", "layer - global", global_coefficient, layer_coefficient, None),
    ]
    OFF = 4.5         # label offset (data units) - sit outside markers
    MARKER_S = 160
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    n = len(metrics)
    ys = list(range(n - 1, -1, -1))
    for y, (label, hint, g, p, ref) in zip(ys, metrics):
        lo, hi = (g, p) if g <= p else (p, g)
        yg, yp = (y + 0.09, y - 0.09) if y == 0 else (y, y)
        ax.plot([g, p], [yg, yp], color=ACCENT, lw=5.0, alpha=0.28,
                solid_capstyle="round", zorder=2)
        ax.scatter([g], [yg], s=MARKER_S, color=MID_SLATE, edgecolor="white",
                   lw=1.6, zorder=5)
        ax.scatter([p], [yp], s=MARKER_S, color=ACCENT, edgecolor="white",
                   lw=1.6, zorder=5)
        if g <= p:
            ax.text(g - OFF, yg, "%.1f" % g, ha="right", va="center",
                    fontsize=9.5, color=MID_SLATE, fontweight="bold")
            ax.text(p + OFF, yp, "%.1f" % p, ha="left", va="center",
                    fontsize=9.5, color=ACCENT, fontweight="bold")
        else:
            ax.text(g + OFF, yg, "%.1f" % g, ha="left", va="center",
                    fontsize=9.5, color=MID_SLATE, fontweight="bold")
            ax.text(p - OFF, yp, "%.1f" % p, ha="right", va="center",
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
