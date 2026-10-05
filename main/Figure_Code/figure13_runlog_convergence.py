# -*- coding: utf-8 -*-
"""
Figure 12 — Run-log search convergence by candidate evaluation, REDESIGNED.
3-panel STACKED layout (one model per row), each panel wider & taller than
the previous side-by-side version, so reviewers can read every datapoint.
Academic styling: subtle alternating stage bands, single-hue model accent,
clean grid, no top/right spines, shared legend above the suptitle.
Output: fig_runlog_convergence_merged.{png,pdf,svg}
"""
import csv
import re
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

RUNLOGS = [
    ("(a) LLaMA3-8B  layer-wise CMA-ES search",  "#5b3b95",
     ROOT / "llama8b" / "逐层搜索" / "layer merge" / "run.log"),
    ("(b) LLaMA3-3B  layer-wise CMA-ES search",  "#2563a8",
     ROOT / "llama3b" / "逐层搜索" / "layer merge" / "run.log"),
    ("(c) SmolLM2 (1.7B)  layer-wise CMA-ES search", "#0e7c66",
     ROOT / "smollm2" / "compare" / "逐层" / "run.log"),
]
MEAN_COLOR = "#c2410c"      # warm orange-red for the per-generation mean
SCATTER_COLOR = "#9aa3ad"
BEST_RING = "#111827"
INK = "#1f2937"
MUTE = "#6b7280"

RESULT_RE = re.compile(r"评估结果.*?总分:\s*([0-9.]+)×[0-9.]+=([0-9.]+)")
SUMMARY_RE = re.compile(r"\[([^\]]+)\]\s*最佳分数:\s*([0-9.]+),\s*平均分数:\s*([0-9.]+)")


def parse_runlog(run_log):
    lines = run_log.read_text(encoding="utf-8", errors="replace").splitlines()
    pending, rows, eidx, gen = [], [], 0, 0
    for line in lines:
        m = RESULT_RE.search(line)
        if m:
            pending.append(float(m.group(2)))
            continue
        s = SUMMARY_RE.search(line)
        if s and pending:
            gen += 1
            stage = s.group(1)
            for ci, score in enumerate(pending, start=1):
                eidx += 1
                rows.append({"Evaluation": eidx, "Generation": gen,
                             "Stage": stage, "Candidate": ci,
                             "Score": round(score, 4)})
            pending = []
    return rows


def draw(ax, title, color, rows):
    x = np.array([r["Evaluation"] for r in rows], dtype=float)
    y = np.array([r["Score"] for r in rows], dtype=float)
    gens = np.array([r["Generation"] for r in rows])
    stages = [r["Stage"] for r in rows]

    gx, gb, gm = [], [], []
    for g in sorted(set(gens)):
        m = gens == g
        gx.append(float(np.mean(x[m])))
        gb.append(float(np.max(y[m])))
        gm.append(float(np.mean(y[m])))

    ymax, ymin = y.max(), y.min()
    yrange = max(ymax - ymin, 0.05)
    ylo = ymin - 0.06 * yrange
    yhi = ymax + 0.30 * yrange
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(0.5, x.max() + 1)

    # alternating stage bands (very subtle)
    band_tones = ["#fafbfc", "#eef2f6"]
    prev = stages[0]
    seg_start = 0.5
    band_idx = 0
    boundaries = []
    for i, st in enumerate(stages):
        if st != prev:
            boundaries.append(i + 0.5)
            ax.axvspan(seg_start, i + 0.5,
                       color=band_tones[band_idx % 2], alpha=0.55, zorder=0)
            band_idx += 1
            seg_start = i + 0.5
            prev = st
    ax.axvspan(seg_start, x.max() + 1,
               color=band_tones[band_idx % 2], alpha=0.55, zorder=0)

    # stage labels in the top margin (within ylim)
    for st in sorted(set(stages), key=stages.index):
        xs = [x[i] for i, v in enumerate(stages) if v == st]
        ax.text(float(np.mean(xs)), yhi - 0.025 * yrange,
                st.replace("_", " "),
                ha="center", va="top", fontsize=9, color="#64748b",
                style="italic")

    # data layers
    ax.scatter(x, y, s=18, color=SCATTER_COLOR, alpha=0.65,
               edgecolors="none", zorder=2)
    ax.plot(gx, gm, color=MEAN_COLOR, lw=1.8, marker="s", ms=4.2, ls="--",
            zorder=3, alpha=0.9)
    ax.plot(gx, gb, color=color, lw=2.6, marker="o", ms=5.5, zorder=4)

    # global best marker
    bi = int(np.argmax(y))
    ax.scatter([x[bi]], [y[bi]], s=140, facecolors="none",
               edgecolors=BEST_RING, lw=1.6, zorder=6)
    near_right = x[bi] > 0.55 * x.max()
    ax.annotate("global best = %.4f" % y[bi], xy=(x[bi], y[bi]),
                xytext=(-14 if near_right else 14, -25),
                textcoords="offset points",
                ha="right" if near_right else "left", fontsize=9,
                color=INK,
                arrowprops={"arrowstyle": "->", "lw": 0.9,
                            "color": "#475569"})

    ax.set_title(title, loc="left", weight="bold", fontsize=11.6, color=INK,
                 pad=8)
    ax.set_xlabel("Number of evaluations", fontsize=10, color=INK)
    ax.set_ylabel("Composite search score", fontsize=10, color=INK)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6, alpha=0.85)
    ax.set_axisbelow(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(labelsize=9, colors=MUTE)


def main():
    plt.rcParams.update({"font.family": "Times New Roman", "axes.linewidth": 0.9,
                         "savefig.dpi": 300})
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 11.6),
                             gridspec_kw={"hspace": 0.45})
    all_rows = []
    for ax, (title, color, log) in zip(axes, RUNLOGS):
        rows = parse_runlog(log)
        if not rows:
            raise SystemExit("no scores parsed from %s" % log)
        draw(ax, title, color, rows)
        for r in rows:
            all_rows.append({"Model": title, **r})
        print("%s: %d evals, %d gens, best %.4f"
              % (title.split()[1], len(rows), rows[-1]["Generation"],
                 max(r["Score"] for r in rows)))

    handles = [
        Line2D([], [], color=SCATTER_COLOR, marker="o", ls="None", ms=6,
               markeredgecolor="none", label="Candidate evaluation"),
        Line2D([], [], color="#444444", marker="o", lw=2.6, ms=6,
               label="Generation best (model colour)"),
        Line2D([], [], color=MEAN_COLOR, marker="s", lw=1.8, ms=5, ls="--",
               label="Generation mean"),
        Line2D([], [], marker="o", color="white",
               markerfacecolor="none", markeredgecolor=BEST_RING, ms=11,
               lw=1.6, label="Global best"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.985),
               ncol=4, frameon=False, fontsize=9.5)
    fig.suptitle("Run-log Search Convergence by Candidate Evaluation",
                 x=0.012, ha="left", weight="bold", fontsize=13.5,
                 color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    stem = OUT / "fig_runlog_convergence_merged"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + ext), bbox_inches="tight",
                    pad_inches=0.07, facecolor="white")
    plt.close(fig)

    with (OUT / "runlog_convergence_merged_data.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["Model", "Evaluation", "Generation",
                                          "Stage", "Candidate", "Score"])
        w.writeheader()
        w.writerows(all_rows)
    print("saved ->", stem)


if __name__ == "__main__":
    main()
