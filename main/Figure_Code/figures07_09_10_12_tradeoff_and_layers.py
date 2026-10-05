from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap


from figure_common import DATA_DIR, output_dir, setup_font, require_file
ROOT = DATA_DIR
OUT = output_dir()


def register_times():
    setup_font()


SMOLLM2_LAYER_DIR = ROOT / "smollm2/compare/layer"

MODEL_ORDER = ["llama3b", "llama8b", "smollm2"]
MODEL_LABELS = {
    "llama3b": "LLaMA3-3B",
    "llama8b": "LLaMA3-8B",
    "smollm2": "SmolLM2",
}
MODEL_COLORS = {
    "llama3b": "#2f6fbb",
    "llama8b": "#6c4fb2",
    "smollm2": "#158466",
}
FORMAL_LAYER_RESULTS = {
    "llama3b": ROOT / "llama3b" / "layer_search" / "layer_merge" / "results.yaml",
    "llama8b": ROOT / "llama8b" / "layer_search" / "layer_merge" / "results.yaml",
    "smollm2": SMOLLM2_LAYER_DIR / "results.yaml",
}
RED = "#d64235"
BLUE = "#2458c7"
GREEN = "#2f8f4e"
GRAY = "#58606a"
LIGHT = "#f6f7f8"
CMAP = LinearSegmentedColormap.from_list("chat_base", ["#3b63d8", "#f7f7f4", "#c71f3a"])


@dataclass
class LayerRun:
    model: str
    label: str
    t_values: np.ndarray
    source: Path
    kind: str
    score: float | None = None


@dataclass
class AblationRun:
    model: str
    label: str
    t: np.ndarray
    asr: np.ndarray
    dialogue: np.ndarray
    source: Path


def configure_style() -> None:
    register_times()
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,

            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def model_from_path(path: Path) -> str | None:
    text = str(path).lower()
    for model in MODEL_ORDER:
        if model in text:
            return model
    return None


def sigmoid_array(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def save_all(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for ext in ["png", "pdf", "svg"]:
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        paths.append(path)
    plt.close(fig)
    return paths


def load_layer_runs() -> tuple[list[LayerRun], list[str]]:
    runs: list[LayerRun] = []
    notes: list[str] = []

    for model, path in FORMAL_LAYER_RESULTS.items():
        if not path.exists():
            notes.append(f"No formal layer coefficients found for {MODEL_LABELS[model]}: {rel(path)}.")
            continue
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        solution = data.get("optimization_history", {}).get("global_best_solution")
        if not solution:
            notes.append(f"{rel(path)} lacks global_best_solution.")
            continue
        runs.append(
            LayerRun(
                model=model,
                label=MODEL_LABELS[model],
                t_values=sigmoid_array(solution),
                source=path,
                kind="formal layer results.yaml",
                score=float(data.get("best_score")) if data.get("best_score") is not None else None,
            )
        )

    selected: list[LayerRun] = []
    for model in MODEL_ORDER:
        matching = [r for r in runs if r.model == model]
        if matching:
            selected.append(matching[0])
        else:
            notes.append(f"No final layer coefficients found for {MODEL_LABELS[model]}.")
    return selected, notes


def load_ablation_runs() -> tuple[list[AblationRun], list[str]]:
    runs: list[AblationRun] = []
    notes: list[str] = []
    for path in sorted(ROOT.glob("*/ablation/*/ablation_results.json")):
        model = model_from_path(path)
        if model is None:
            notes.append(f"Could not infer model for {rel(path)}.")
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("results", [])
        if not rows:
            notes.append(f"{rel(path)} has no ablation results.")
            continue
        runs.append(
            AblationRun(
                model=model,
                label=MODEL_LABELS[model],
                t=np.array([r["t"] for r in rows], dtype=float),
                asr=np.array([r["asr"] for r in rows], dtype=float),
                dialogue=np.array([r["dialogue"] for r in rows], dtype=float),
                source=path,
            )
        )
    for model in MODEL_ORDER:
        if not any(r.model == model for r in runs):
            notes.append(f"No ablation JSON found for {MODEL_LABELS[model]}.")
    return runs, notes


def fig4_layer_coefficients_overview(layer_runs: list[LayerRun]) -> list[Path]:
    fig = plt.figure(figsize=(11.8, 7.0))
    gs = gridspec.GridSpec(3, 2, width_ratios=[3.8, 1.05], hspace=0.42, wspace=0.18)

    for i, run in enumerate(layer_runs):
        layers = np.arange(len(run.t_values))
        ax = fig.add_subplot(gs[i, 0])
        color = MODEL_COLORS[run.model]
        ax.axhspan(0.5, 1.0, color="#f9dfdc", alpha=0.55, zorder=0)
        ax.axhspan(0.0, 0.5, color="#dce6fb", alpha=0.6, zorder=0)
        ax.axhline(0.5, color="#555555", lw=0.9, ls="--")
        ax.plot(layers, run.t_values, color=color, lw=2.0, marker="o", ms=3.2)
        ax.fill_between(layers, 0.5, run.t_values, where=run.t_values >= 0.5, color=RED, alpha=0.12, interpolate=True)
        ax.fill_between(layers, run.t_values, 0.5, where=run.t_values < 0.5, color=BLUE, alpha=0.12, interpolate=True)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(run.t_values) - 0.5)
        ax.set_ylabel("Base weight t")
        ax.set_title(f"{run.label}: per-layer_merge coefficients", loc="left", weight="bold")
        if i == len(layer_runs) - 1:
            ax.set_xlabel("Layer index")
        else:
            ax.set_xticklabels([])
        ax.grid(axis="y", color="#d8d8d8", lw=0.6, alpha=0.8)
        ax.text(len(run.t_values) - 0.2, 0.93, "base-dominant", ha="right", va="top", fontsize=7.5, color=RED)
        ax.text(len(run.t_values) - 0.2, 0.07, "chat-dominant", ha="right", va="bottom", fontsize=7.5, color=BLUE)

        hax = fig.add_subplot(gs[i, 1])
        hax.imshow(run.t_values.reshape(1, -1), aspect="auto", cmap=CMAP, vmin=0, vmax=1)
        hax.set_yticks([])
        tick_locs = np.linspace(0, len(run.t_values) - 1, min(5, len(run.t_values)), dtype=int)
        hax.set_xticks(tick_locs)
        hax.set_xticklabels([str(t) for t in tick_locs])
        hax.set_title("Heat strip", fontsize=9)
        for spine in hax.spines.values():
            spine.set_visible(False)

    cax = fig.add_axes([0.91, 0.17, 0.014, 0.66])
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=CMAP, norm=norm)
    cb.set_label("Base weight t", rotation=90)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle("Layer-wise merge coefficients across model scales", x=0.05, ha="left", fontsize=14, weight="bold")
    return save_all(fig, "figure09_layer_coefficients")


def fig5_ablation_tradeoff(ablation_runs: list[AblationRun]) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.5), sharex=True, sharey="row")
    for col, model in enumerate(MODEL_ORDER):
        runs = [r for r in ablation_runs if r.model == model]
        for row, metric in enumerate(["asr", "dialogue"]):
            ax = axes[row, col]
            color = RED if metric == "asr" else BLUE
            for run in runs:
                y = run.asr if metric == "asr" else run.dialogue
                ax.plot(run.t, y, color=color, alpha=0.28, lw=1.8, marker="o", ms=3)
            if runs:
                xs = runs[0].t
                stacked = np.vstack([r.asr if metric == "asr" else r.dialogue for r in runs])
                ax.plot(xs, stacked.mean(axis=0), color=color, lw=2.8, marker="o", ms=4.5, label="mean" if len(runs) > 1 else "run")
                if len(runs) > 1:
                    ax.fill_between(xs, stacked.min(axis=0), stacked.max(axis=0), color=color, alpha=0.10, lw=0)
            else:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", color=GRAY)
            ax.set_title(f"{MODEL_LABELS[model]} (n={len(runs)})", weight="bold")
            ax.set_ylim(0, 1)
            ax.set_xlim(0.05, 0.95)
            ax.grid(color="#dcdcdc", lw=0.6, alpha=0.75)
            if col == 0:
                ax.set_ylabel("ASR / harmful rate" if metric == "asr" else "Dialogue quality")
            if row == 1:
                ax.set_xlabel("Base model weight t")
    fig.suptitle("Ablation trade-off under fixed merge ratios", x=0.05, ha="left", fontsize=14, weight="bold")
    fig.text(0.05, 0.925, "Repeated runs are translucent; bold curves show the available mean.", color=GRAY, fontsize=9)
    return save_all(fig, "figure07_mixing_tradeoff")


def load_formal_layer_curve(path: Path) -> dict | None:
    if not path.exists():
        return None
    model = model_from_path(path)
    log_path = path.with_name("run.log")
    if log_path.exists():
        score_pattern = re.compile(r"总分:\s*[-+0-9.]+×[-+0-9.]+\s*=\s*([-+0-9.]+)")
        stage_pattern = re.compile(r"^---\s*(.*?)\s*/\s*第\s*(\d+)\s*代\s*---")
        scores: list[float] = []
        stages: list[str] = []
        current_stage = ""
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stage_match = stage_pattern.search(line)
            if stage_match:
                current_stage = stage_match.group(1)
                continue
            score_match = score_pattern.search(line)
            if score_match:
                scores.append(float(score_match.group(1)))
                stages.append(current_stage)
        if scores:
            return {
                "model": model,
                "label": MODEL_LABELS.get(model or "", path.parent.name),
                "x": np.arange(1, len(scores) + 1, dtype=float),
                "score": np.array(scores, dtype=float),
                "stages": stages,
                "source": log_path,
                "kind": "formal run.log evaluation scores",
            }

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    hist = data.get("optimization_history", {})
    scores = hist.get("best_scores")
    if not scores:
        return None
    generations = hist.get("generations") or list(range(1, len(scores) + 1))
    return {
        "model": model,
        "label": MODEL_LABELS.get(model or "", path.parent.name),
        "x": np.array(generations, dtype=float),
        "score": np.array(scores, dtype=float),
        "source": path,
        "kind": "fallback formal results.yaml generation scores",
    }


def figA4_formal_layer_convergence() -> tuple[list[Path], list[str]]:
    notes: list[str] = []
    formal_curves = [c for p in FORMAL_LAYER_RESULTS.values() if (c := load_formal_layer_curve(p))]
    for model, path in FORMAL_LAYER_RESULTS.items():
        if not path.exists():
            notes.append(f"No formal layer results.yaml found for {MODEL_LABELS[model]}: {rel(path)}.")
    if formal_curves:
        notes.append("Formal convergence now uses evaluation-level composite scores parsed from formal `run.log` files when available; `results.yaml` generation scores are only a fallback.")

    fig_width = max(12.0, 4.4 * len(FORMAL_LAYER_RESULTS))
    fig, axes = plt.subplots(1, len(FORMAL_LAYER_RESULTS), figsize=(fig_width, 4.2), sharey=False)
    if len(FORMAL_LAYER_RESULTS) == 1:
        axes = [axes]

    curve_by_model = {curve["model"]: curve for curve in formal_curves}
    for ax, (model, path) in zip(axes, FORMAL_LAYER_RESULTS.items()):
        curve = curve_by_model.get(model)
        color = MODEL_COLORS.get(model, GRAY)
        if curve is None:
            ax.text(0.5, 0.5, "missing formal layer score history", transform=ax.transAxes, ha="center", va="center", color=GRAY)
        else:
            stage_positions: list[int] = []
            prev_stage = curve.get("stages", [None])[0] if curve.get("stages") else None
            for idx, stage in enumerate(curve.get("stages", []), start=1):
                if idx > 1 and stage != prev_stage:
                    stage_positions.append(idx)
                    prev_stage = stage
            ax.plot(
                curve["x"],
                curve["score"],
                color=color,
                lw=2.4,
                marker="o",
                ms=2.8,
                label="Evaluation score",
            )
            finite_scores = curve["score"][np.isfinite(curve["score"])]
            if finite_scores.size:
                ymin = float(np.min(finite_scores))
                ymax = float(np.max(finite_scores))
                yrange = max(ymax - ymin, 0.01)
                pad = yrange * 0.16
                ax.set_ylim(ymin - pad, ymax + pad)
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6))
                ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
            for pos in stage_positions:
                ax.axvline(pos, color="#999999", lw=0.8, ls="--", alpha=0.45)
        ax.set_title(f"{MODEL_LABELS[model]} formal layer search", weight="bold", loc="left")
        ax.set_xlabel("Evaluation index")
        ax.set_ylabel("Composite score")
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(color="#dcdcdc", lw=0.6, alpha=0.75)
        ax.legend(frameon=False, loc="best", fontsize=8)
    fig.suptitle("Formal layer-search convergence by evaluation", x=0.05, ha="left", fontsize=14, weight="bold")
    fig.text(0.05, 0.91, "Each point is one evaluated candidate from the formal layer-search run log; dashed vertical lines mark stage transitions.", color=GRAY, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.15, wspace=0.20)
    return save_all(fig, "figure10_formal_convergence"), notes


def split_segments(n_layers: int) -> list[tuple[str, np.ndarray]]:
    indices = np.arange(n_layers)
    chunks = np.array_split(indices, 3)
    return [("early", chunks[0]), ("middle", chunks[1]), ("late", chunks[2])]


def figA5_layer_segment_stats(layer_runs: list[LayerRun], top_k: int = 5) -> list[Path]:
    fig = plt.figure(figsize=(12.4, 6.8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.25)
    ax_mean = fig.add_subplot(gs[0, 0])
    ax_frac = fig.add_subplot(gs[0, 1])
    ax_top = fig.add_subplot(gs[1, :])

    seg_labels = ["early", "middle", "late"]
    x = np.arange(len(seg_labels))
    width = 0.22
    for i, run in enumerate(layer_runs):
        means = []
        fracs = []
        for _, idx in split_segments(len(run.t_values)):
            values = run.t_values[idx]
            means.append(float(values.mean()))
            fracs.append(float((values > 0.5).mean()))
        offset = (i - (len(layer_runs) - 1) / 2) * width
        ax_mean.bar(x + offset, means, width, label=run.label, color=MODEL_COLORS[run.model], alpha=0.86)
        ax_frac.bar(x + offset, fracs, width, label=run.label, color=MODEL_COLORS[run.model], alpha=0.86)

    for ax, ylabel, title in [
        (ax_mean, "Mean base weight t", "Segment-level average coefficients"),
        (ax_frac, "Fraction of layers with t > 0.5", "Base-dominant layer share"),
    ]:
        ax.axhline(0.5, color="#555555", lw=0.9, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(seg_labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.set_title(title, weight="bold", loc="left")
        ax.grid(axis="y", color="#dcdcdc", lw=0.6, alpha=0.75)
    ax_frac.legend(frameon=False, loc="upper right")

    rows = []
    colors = []
    for run in layer_runs:
        order = np.argsort(np.abs(run.t_values - 0.5))[::-1][:top_k]
        signed = []
        row_colors = []
        for idx in order:
            val = run.t_values[idx]
            direction = "B" if val > 0.5 else "C"
            signed.append(f"L{idx}: {val:.2f} {direction}")
            row_colors.append(CMAP(val))
        rows.append(signed)
        colors.append(row_colors)

    ax_top.set_axis_off()
    table = ax_top.table(
        cellText=rows,
        rowLabels=[r.label for r in layer_runs],
        colLabels=[f"Top {i + 1}" for i in range(top_k)],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)
    for row_i, run in enumerate(layer_runs, start=1):
        label_cell = table[(row_i, -1)]
        label_cell.set_text_props(weight="bold", color=MODEL_COLORS[run.model])
        for col_i in range(top_k):
            cell = table[(row_i, col_i)]
            cell.set_facecolor(colors[row_i - 1][col_i])
            cell.set_alpha(0.22)
    for col_i in range(top_k):
        table[(0, col_i)].set_facecolor("#eef0f3")
        table[(0, col_i)].set_text_props(weight="bold")
    ax_top.set_title("Most extreme layers by |t - 0.5| (B=base-dominant, C=chat-dominant)", weight="bold", loc="left", pad=10)

    fig.suptitle("Layer segment statistics and extreme layers", x=0.05, ha="left", fontsize=14, weight="bold")
    return save_all(fig, "figure12_layer_statistics")


def main():
    configure_style()
    for path in FORMAL_LAYER_RESULTS.values():
        require_file(path)
        require_file(path.with_name("run.log"))
    layer_runs, notes = load_layer_runs()
    ablation_runs, ablation_notes = load_ablation_runs()
    if len(layer_runs) != 3 or any(not any(r.model == m for r in ablation_runs) for m in MODEL_ORDER):
        raise ValueError("Missing layer or ablation inputs: " + "; ".join(notes + ablation_notes))
    if any(load_formal_layer_curve(p) is None for p in FORMAL_LAYER_RESULTS.values()):
        raise ValueError("No formal convergence scores in one or more input files")
    fig4_layer_coefficients_overview(layer_runs)
    fig5_ablation_tradeoff(ablation_runs)
    figA4_formal_layer_convergence()
    figA5_layer_segment_stats(layer_runs)
    print("Figures 7, 9, 10 and 12 saved to", OUT)

if __name__ == "__main__":
    main()
