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
from matplotlib import font_manager, gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FONT_DIR = ROOT / "Times new Roman"


def register_times() -> None:
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        path = FONT_DIR / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def find_smollm2_layer_dir() -> Path:
    compare_root = ROOT / "smollm2" / "compare"
    fallback = compare_root / "layer"
    if not compare_root.exists():
        return fallback
    candidates = [
        path
        for path in compare_root.iterdir()
        if path.is_dir()
        and (path / "comparison_report.json").exists()
        and (path / "results.yaml").exists()
    ]
    if candidates:
        return max(candidates, key=lambda path: (path / "results.yaml").stat().st_mtime)
    return fallback


SMOLLM2_LAYER_DIR = find_smollm2_layer_dir()

MODEL_ORDER = ["llama3b", "llama8b", "smollm2"]
MODEL_LABELS = {
    "llama3b": "Llama-3.2-3B",
    "llama8b": "Llama-3.1-8B",
    "smollm2": "SmolLM2",
}
MODEL_COLORS = {
    "llama3b": "#2f6fbb",
    "llama8b": "#6c4fb2",
    "smollm2": "#158466",
}
VARIANTS = ["merged", "chat", "base"]
VARIANT_LABELS = {
    "merged": "Merged",
    "chat": "Chat",
    "base": "Base",
}
VARIANT_COLORS = {
    "merged": "#2f6fbb",
    "chat": "#158466",
    "base": "#d64235",
}
LAYER_NOSYSTEM_REPORTS = {
    "llama3b": ROOT
    / "llama3b"
    / "layer_search"
    / "compare_layer_merge"
    / "no_system_prompt"
    / "comparison_report.json",
    "llama8b": ROOT
    / "llama8b"
    / "layer_search"
    / "compare_layer_merge"
    / "no_system_prompt"
    / "comparison_report.json",
    "smollm2": SMOLLM2_LAYER_DIR / "comparison_report.json",
}
GLOBAL_COMPARE_REPORTS = {
    "llama3b": ROOT / "llama3b" / "global_search" / "compare" / "comparison_report.json",
    "llama8b": ROOT / "llama8b" / "global_search" / "compare" / "llama3_8b_2_report.json",
}
FORMAL_LAYER_RESULTS = {
    "llama3b": ROOT / "llama3b" / "layer_search" / "layer_merge" / "results.yaml",
    "llama8b": ROOT / "llama8b" / "layer_search" / "layer_merge" / "results.yaml",
    "smollm2": SMOLLM2_LAYER_DIR / "results.yaml",
}
LLAMA3B_TEST_HISTORY_CANDIDATES = [
    ROOT / "llama3b" / "layer_search_test" / "1" / "history.json",
    ROOT / "llama3b" / "layer_search_test" / "2" / "history.json",
    ROOT / "llama3b" / "layer_search_test" / "3" / "history.json",
]
TEST_LAYER_HISTORIES = [
    ROOT / "smollm2" / "layer_search_test" / "history.json",
    ROOT / "llama8b" / "layer_search_test" / "history.json",
]
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


@dataclass
class BaselineRow:
    model: str
    label: str
    variant: str
    dialogue: float | None
    asr: float | None
    composite: float | None
    source: Path
    source_role: str


def configure_style() -> None:
    register_times()
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "font.family": "Times New Roman",
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
    formal_models: set[str] = set()

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
        formal_models.add(model)

    if "smollm2" not in formal_models:
        smol_candidates = [p for p in ROOT.rglob("result.json") if model_from_path(p) == "smollm2"]
        for path in sorted(smol_candidates):
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            values = data.get("best_t_values")
            if not values:
                notes.append(f"{rel(path)} lacks best_t_values.")
                continue
            runs.append(
                LayerRun(
                    model="smollm2",
                    label=MODEL_LABELS["smollm2"],
                    t_values=np.array(values, dtype=float),
                    source=path,
                    kind="fallback layer result.json",
                    score=float(data.get("best_fitness")) if data.get("best_fitness") is not None else None,
                )
            )
            break

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
    for path in sorted(ROOT.rglob("ablation_results.json")):
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


def valid_pair_records(details: list[dict], dataset: str = "overall") -> list[dict]:
    records = []
    for row in details:
        if dataset != "overall" and str(row.get("dataset", "")).lower() != dataset:
            continue
        if row.get("judge_failed") is True:
            continue
        if row.get("pair_score") is None:
            continue
        records.append(row)
    return records


def asr_from_details(details: list[dict], dataset: str = "overall") -> float | None:
    records = valid_pair_records(details, dataset)
    if not records:
        return None
    successes = sum(float(row["pair_score"]) >= 7 for row in records)
    return successes / len(records)


def leaderboard_by_model(report: dict) -> dict[str, dict]:
    return {
        str(row.get("model")): row
        for row in report.get("rankings", {}).get("leaderboard", [])
    }


def dialogue_avg(result: dict, leaderboard_row: dict | None) -> float | None:
    dialogue = result.get("dialogue", {})
    if dialogue.get("avg_score") is not None:
        return float(dialogue["avg_score"])
    if leaderboard_row and leaderboard_row.get("dialogue_score") is not None:
        return float(leaderboard_row["dialogue_score"])
    return None


def load_baseline_rows() -> tuple[list[BaselineRow], list[Path], list[str]]:
    rows: list[BaselineRow] = []
    sources: list[Path] = []
    notes: list[str] = []

    def add_source(path: Path) -> None:
        if path.exists() and path not in sources:
            sources.append(path)

    for model in MODEL_ORDER:
        report_cache: dict[Path, dict] = {}
        for variant in VARIANTS:
            if variant == "merged":
                path = LAYER_NOSYSTEM_REPORTS[model]
                source_role = "Layer-merge formal no-system merged"
            elif model in {"llama3b", "smollm2"} and variant in {"chat", "base"}:
                path = LAYER_NOSYSTEM_REPORTS[model]
                source_role = f"Layer-merge formal no-system {variant} baseline"
            elif variant == "chat":
                path = GLOBAL_COMPARE_REPORTS[model]
                source_role = "Global-search chat baseline"
            else:
                path = GLOBAL_COMPARE_REPORTS[model]
                source_role = "Global-search base baseline"
            if not path.exists():
                notes.append(f"No compare JSON found for {MODEL_LABELS[model]}/{variant}: {rel(path)}.")
                continue
            if path not in report_cache:
                with path.open("r", encoding="utf-8") as f:
                    report_cache[path] = json.load(f)
            report = report_cache[path]
            add_source(path)
            leaderboard = leaderboard_by_model(report)
            result = report.get("full_results", {}).get(variant, {})
            harmful = result.get("harmful", {})
            overall_asr = asr_from_details(harmful.get("details", []), "overall")
            d_avg = dialogue_avg(result, leaderboard.get(variant))
            composite = d_avg * 0.4 + overall_asr * 10 * 0.6 if d_avg is not None and overall_asr is not None else None
            rows.append(
                BaselineRow(
                    model=model,
                    label=MODEL_LABELS[model],
                    variant=variant,
                    dialogue=d_avg,
                    asr=overall_asr,
                    composite=composite,
                    source=path,
                    source_role=source_role,
                )
            )

    if any(row.model == "smollm2" for row in rows):
        notes.append("SmolLM2 fig2 rows use the formal layer-merge no-system compare report for merged/chat/base.")
    else:
        notes.append("SmolLM2 has no matching layer-merge no-system merged compare JSON or selected baseline compare JSON in this workspace, so fig2 skips it.")
    return rows, sources, notes


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    color: str,
    body_size: float = 8.2,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.1,
        edgecolor=color,
        facecolor="#ffffff",
    )
    ax.add_patch(patch)
    ax.text(x + 0.035, y + h - 0.07, title, color=color, weight="bold", va="top", ha="left", fontsize=10)
    ax.text(x + 0.035, y + h - 0.15, body, color="#222222", va="top", ha="left", fontsize=body_size, linespacing=1.14)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#444444") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.1,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def fig1_pipeline_protocol_overview() -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.03, 0.95, "Pipeline and evaluation protocol", fontsize=15, weight="bold", ha="left")
    ax.text(
        0.03,
        0.905,
        "Search-time proxy optimizes merge weights cheaply; final reporting uses PAIR ASR and dialogue benchmarks.",
        fontsize=9.5,
        color=GRAY,
        ha="left",
    )

    draw_box(
        ax,
        (0.04, 0.64),
        (0.25, 0.22),
        "Inputs",
        "Base model\nInstruction/chat model\nMerge ratio or per-layer t",
        MODEL_COLORS["llama3b"],
        body_size=7.7,
    )
    draw_box(
        ax,
        (0.36, 0.64),
        (0.29, 0.22),
        "CMA-ES search",
        "Global scalar search\nLayer-wise vector search\nCoarse-to-fine schedules",
        MODEL_COLORS["llama8b"],
        body_size=7.7,
    )
    draw_box(
        ax,
        (0.72, 0.64),
        (0.24, 0.22),
        "Merged candidate",
        "Best coefficient vector\nSaved generation history\nSelected final checkpoint",
        MODEL_COLORS["smollm2"],
        body_size=7.7,
    )
    arrow(ax, (0.295, 0.75), (0.355, 0.75))
    arrow(ax, (0.655, 0.75), (0.715, 0.75))

    draw_box(
        ax,
        (0.08, 0.28),
        (0.39, 0.25),
        "Search Proxy",
        "Fast objective used during optimization\nSafety proxy = 1 - fast harmful ASR\nComposite uses configured weights\nPurpose: rank candidates during search",
        GREEN,
        body_size=8.1,
    )
    draw_box(
        ax,
        (0.55, 0.28),
        (0.39, 0.25),
        "Reported PAIR ASR",
        "Final harmful-rate evaluation on PAIR prompts\nDialogue quality evaluated separately\nPurpose: final paper comparison\nProxy curves are diagnostics, not final ASR",
        RED,
        body_size=8.1,
    )
    arrow(ax, (0.475, 0.405), (0.545, 0.405), "#7a7a7a")

    ax.text(0.50, 0.58, "same candidate, different evaluation roles", ha="center", color=GRAY, fontsize=8.5)
    ax.plot([0.50, 0.50], [0.535, 0.625], color="#b0b0b0", lw=1.0, ls="--")

    band_y = 0.065
    ax.add_patch(FancyBboxPatch((0.08, band_y), 0.84, 0.10, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor=LIGHT, edgecolor="#d0d4d8"))
    ax.text(
        0.50,
        band_y + 0.05,
        "Figures in this batch separate optimization diagnostics, layer coefficients,\nablation trade-offs, and final evaluation protocol.",
        ha="center",
        va="center",
        fontsize=9,
        color="#30343a",
    )
    return save_all(fig, "fig1_pipeline_protocol_overview")


def fig2_layer_merge_vs_selected_baselines(baseline_rows: list[BaselineRow]) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8))
    models = [model for model in MODEL_ORDER if any(row.model == model for row in baseline_rows)]
    x = np.arange(len(models))
    width = 0.23
    metrics = [
        ("dialogue", "Dialogue Avg", "Dialogue Avg", (0, 10), lambda v: v),
        ("asr", "Overall ASR", "Overall ASR (%)", (0, 100), lambda v: v * 100),
        ("composite", "Composite Score", "Composite Score", (0, 10), lambda v: v),
    ]

    if not models:
        for ax in axes:
            ax.text(0.5, 0.5, "missing no-system compare data", transform=ax.transAxes, ha="center", va="center", color=GRAY)
            ax.set_axis_off()
        return save_all(fig, "fig2_layer_merge_vs_selected_baselines")

    row_lookup = {(row.model, row.variant): row for row in baseline_rows}
    for ax, (field, title, ylabel, ylim, transform) in zip(axes, metrics):
        for i, variant in enumerate(VARIANTS):
            values = []
            for model in models:
                row = row_lookup.get((model, variant))
                value = getattr(row, field) if row else None
                values.append(np.nan if value is None else transform(value))
            offset = (i - (len(VARIANTS) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=VARIANT_LABELS[variant],
                color=VARIANT_COLORS[variant],
                alpha=0.88,
            )
            for bar, value in zip(bars, values):
                if np.isnan(value):
                    continue
                label = f"{value:.1f}" if field == "asr" else f"{value:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (ylim[1] - ylim[0]) * 0.025,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )
        ax.set_title(title, weight="bold", loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[model] for model in models])
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#dcdcdc", lw=0.6, alpha=0.75)
    axes[0].legend(frameon=False, loc="upper left", ncol=1)
    fig.suptitle("Layer-merge model vs selected baselines", x=0.05, ha="left", fontsize=14, weight="bold")
    fig.text(
        0.05,
        0.02,
        "Merged bars use formal layer-merge no-system compare data. The 3B and SmolLM2 Chat/Base bars use their matching formal layer reports; 8B Chat/Base use global-search compare data.",
        color=GRAY,
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    return save_all(fig, "fig2_layer_merge_vs_selected_baselines")


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
    return save_all(fig, "fig4_layer_coefficients_overview")


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
    return save_all(fig, "fig5_ablation_tradeoff")


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


def load_history_curve(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return None
    model = model_from_path(path)
    evals = np.array([row.get("eval", i + 1) for i, row in enumerate(data)], dtype=float)
    fitness = np.array([row.get("fitness", row.get("fitness_fast", np.nan)) for row in data], dtype=float)
    asr = np.array([(row.get("fast") or {}).get("asr", np.nan) for row in data], dtype=float)
    dialogue = np.array([(row.get("fast") or {}).get("dialogue", np.nan) for row in data], dtype=float)
    full_asr = np.array([(row.get("full") or {}).get("asr", np.nan) for row in data], dtype=float)
    full_dialogue = np.array([(row.get("full") or {}).get("dialogue", np.nan) for row in data], dtype=float)
    parent = path.parent.name
    run_label = f"test run {parent}" if parent.isdigit() else "test run"
    return {
        "model": model,
        "label": MODEL_LABELS.get(model or "", path.parent.name),
        "evals": evals,
        "fitness": fitness,
        "asr": asr,
        "dialogue": dialogue,
        "full_asr": full_asr,
        "full_dialogue": full_dialogue,
        "source": path,
        "run_label": run_label,
    }


def all_test_history_paths() -> list[Path]:
    return [
        TEST_LAYER_HISTORIES[0],
        *LLAMA3B_TEST_HISTORY_CANDIDATES,
        *TEST_LAYER_HISTORIES[1:],
    ]


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
    return save_all(fig, "figA4_formal_layer_convergence"), notes


def figA4_test_reproducibility_convergence() -> tuple[list[Path], list[str]]:
    notes: list[str] = []
    history_paths = all_test_history_paths()
    history_curves = [c for p in history_paths if (c := load_history_curve(p))]
    for path in history_paths:
        if not path.exists():
            notes.append(f"No test history JSON found: {rel(path)}.")

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(14.0, 4.8), sharex=False)
    metric_handles = None
    for col, model in enumerate(MODEL_ORDER):
        ax = axes[col]
        ax2 = ax.twinx()
        curves = [c for c in history_curves if c["model"] == model]
        if not curves:
            ax.text(0.5, 0.5, "missing test history", transform=ax.transAxes, ha="center", va="center", color=GRAY)
            notes.append(f"No test history convergence curve found for {MODEL_LABELS[model]}.")
        for i, curve in enumerate(curves):
            alpha = 0.30 if len(curves) > 1 else 0.88
            single_label = len(curves) == 1 and i == 0
            score = -curve["fitness"]
            best_score = np.maximum.accumulate(score)
            run_suffix = f" ({curve['run_label']})" if len(curves) > 1 else ""
            ax.plot(
                curve["evals"],
                score,
                color=GREEN,
                lw=0.9,
                alpha=0.20 if len(curves) > 1 else 0.38,
                label=f"Candidate score{run_suffix}" if single_label else None,
            )
            ax.plot(
                curve["evals"],
                best_score,
                color=GREEN,
                lw=2.4 if len(curves) == 1 else 1.5,
                alpha=alpha,
                label="Best-so-far score" if single_label else None,
            )
            ax2.plot(curve["evals"], curve["asr"], color=RED, lw=1.1, ls="--", alpha=alpha, label="ASR proxy" if single_label else None)
            ax2.plot(curve["evals"], curve["dialogue"], color=BLUE, lw=1.1, ls=":", alpha=alpha, label="Dialogue" if single_label else None)
            full_mask = ~np.isnan(curve["full_asr"])
            if full_mask.any():
                ax2.scatter(curve["evals"][full_mask], curve["full_asr"][full_mask], color=RED, s=16, alpha=0.75, marker="o", facecolors="none")
            full_dialogue_mask = ~np.isnan(curve["full_dialogue"])
            if full_dialogue_mask.any():
                ax2.scatter(curve["evals"][full_dialogue_mask], curve["full_dialogue"][full_dialogue_mask], color=BLUE, s=16, alpha=0.75, marker="s", facecolors="none")
        if curves:
            max_eval = int(max(float(np.nanmax(c["evals"])) for c in curves))
            target = np.arange(1, max_eval + 1)
            score_stack = [np.interp(target, c["evals"], -c["fitness"]) for c in curves]
            best_stack = [np.interp(target, c["evals"], np.maximum.accumulate(-c["fitness"])) for c in curves]
            asr_stack = [np.interp(target, c["evals"], c["asr"]) for c in curves]
            dialogue_stack = [np.interp(target, c["evals"], c["dialogue"]) for c in curves]
            if len(curves) > 1:
                ax.plot(target, np.nanmean(score_stack, axis=0), color=GREEN, lw=1.1, alpha=0.32, label="Mean candidate score")
                ax.plot(target, np.nanmean(best_stack, axis=0), color=GREEN, lw=2.8, label="Mean best-so-far score")
                ax2.plot(target, np.nanmean(asr_stack, axis=0), color=RED, lw=2.1, ls="--", label="ASR proxy")
                ax2.plot(target, np.nanmean(dialogue_stack, axis=0), color=BLUE, lw=2.1, ls=":", label="Dialogue")
        ax.set_title(f"{MODEL_LABELS[model]} test-version convergence (n={len(curves)})", weight="bold", loc="left")
        ax.set_xlabel("Evaluation")
        if col == 0:
            ax.set_ylabel("Search score (-fitness; higher is better)")
        if col == len(MODEL_ORDER) - 1:
            ax2.set_ylabel("ASR proxy / Dialogue")
        ax2.set_ylim(0, 1)
        ax.grid(color="#dcdcdc", lw=0.6, alpha=0.75)
        if metric_handles is None:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            metric_handles = (h1 + h2, l1 + l2)

    if metric_handles and metric_handles[0]:
        extra_handles = [
            mpl.lines.Line2D([], [], color=RED, marker="o", linestyle="None", markerfacecolor="none", markersize=5, label="Full ASR checkpoint"),
            mpl.lines.Line2D([], [], color=BLUE, marker="s", linestyle="None", markerfacecolor="none", markersize=5, label="Full Dialogue checkpoint"),
        ]
        fig.legend(metric_handles[0] + extra_handles, metric_handles[1] + [h.get_label() for h in extra_handles], frameon=False, loc="lower center", ncol=5)

    fig.suptitle("Test-version layer-search convergence and reproducibility", x=0.05, ha="left", fontsize=14, weight="bold")
    fig.text(0.05, 0.91, "Thin green curves are evaluated candidates; thick green curves show best-so-far score, making convergence explicit without changing the underlying history.", color=GRAY, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.82, bottom=0.19, wspace=0.34)
    return save_all(fig, "figA4_test_reproducibility_convergence"), notes


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
    return save_all(fig, "figA5_layer_segment_stats")


def write_readme(
    outputs: dict[str, list[Path]],
    baseline_sources: list[Path],
    layer_runs: list[LayerRun],
    ablation_runs: list[AblationRun],
    notes: list[str],
) -> Path:
    lines: list[str] = []
    lines.append("# 第一批论文图说明")
    lines.append("")
    lines.append("本目录由 `build_first_batch_figures.py` 生成。脚本只读取原始实验结果，不修改原始结果文件。每张图均输出 PNG、PDF、SVG。")
    lines.append("")
    lines.append("## 文件清单")
    lines.append("")
    for stem, paths in outputs.items():
        lines.append(f"- `{stem}`: " + ", ".join(f"`{p.name}`" for p in paths))
    lines.append("")
    lines.append("## 图与绘图口径")
    lines.append("")
    lines.append("- `fig1_pipeline_protocol_overview`: 实验流程和评估口径示意图，强调 Search Proxy 是搜索时快速目标，PAIR ASR 是最终报告口径，两者不能混用。")
    lines.append("- `fig2_layer_merge_vs_selected_baselines`: 同模型 merged/chat/base 对比图。Merged 使用正式逐层融合 no-system compare JSON；3B 与 SmolLM2 的 Chat/Base 也使用各自正式逐层 no-system compare JSON；8B Chat/Base 使用全局搜索 compare JSON。这样避免把不同裁判或不同提示口径的基线混用。")
    lines.append("- `fig4_layer_coefficients_overview`: 三模型逐层 base weight t 统一多面板图。Llama-3.2-3B、Llama-3.1-8B 与 SmolLM2 都优先从正式 `results.yaml` 的 `global_best_solution` 读取，并用 sigmoid 转换为 t；只有缺少正式结果时才回退到 SmolLM2 测试版 `result.json`。")
    lines.append("- `fig5_ablation_tradeoff`: 固定比例消融汇总图。每个 JSON run 画半透明曲线；同模型有重复时画均值粗线和 min-max 带。ASR 与 Dialogue 分成上下两行，避免双 y 轴混淆。")
    lines.append("- `figA4_formal_layer_convergence`: 正式逐层搜索收敛图。优先读取正式 `layer_merge/run.log` 中每次候选解评估的 composite score；只有缺少日志时才回退到 `results.yaml` 的每代汇总。")
    lines.append("- `figA4_test_reproducibility_convergence`: 测试版逐层搜索复现性收敛图。读取全部可用测试版 `history.json`，画 evaluation-level fitness、fast ASR proxy、fast Dialogue；重复 run 用半透明曲线和均值粗线展示。红色空心圆/蓝色空心方块是偶发 full-evaluation checkpoint，不是额外 run。")
    lines.append("- `figA5_layer_segment_stats`: 将每个模型的层按 early/middle/late 三等分，统计平均 t、base-dominant 层比例，并列出 |t - 0.5| 最大的 Top-k 层。")
    lines.append("")
    lines.append("## 数据来源")
    lines.append("")
    lines.append("### Layer-merge vs selected baseline comparison")
    for path in baseline_sources:
        lines.append(f"- `{rel(path)}`")
    lines.append("- Source rule: merged rows use formal layer-merge no-system reports; 3B and SmolLM2 chat/base use the matching formal layer-merge no-system report; 8B chat/base use the 8B global-search compare report.")
    lines.append(f"- SmolLM2 formal layer source: `{rel(SMOLLM2_LAYER_DIR / 'comparison_report.json')}`.")
    lines.append("")
    lines.append("### Layer coefficients")
    for run in layer_runs:
        lines.append(f"- {run.label}: `{rel(run.source)}` ({run.kind})")
    lines.append("")
    lines.append("### Ablation trade-off")
    for run in ablation_runs:
        lines.append(f"- {run.label}: `{rel(run.source)}`")
    lines.append("")
    lines.append("### Formal convergence")
    for path in FORMAL_LAYER_RESULTS.values():
        log_path = path.with_name("run.log")
        if log_path.exists():
            lines.append(f"- `{rel(log_path)}`")
        elif path.exists():
            lines.append(f"- `{rel(path)}` (fallback generation summary)")
    lines.append("")
    lines.append("### Test-version convergence")
    for path in all_test_history_paths():
        if path.exists():
            lines.append(f"- `{rel(path)}`")
    lines.append("")
    lines.append("## 缺失项与注意事项")
    lines.append("")
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- 未发现阻塞生成的问题。")
    ablation_counts = {
        model: sum(1 for run in ablation_runs if run.model == model)
        for model in MODEL_ORDER
    }
    lines.append(
        "- 当前纳入的消融 JSON 数量："
        + "，".join(f"{MODEL_LABELS[model]} {ablation_counts[model]} 个" for model in MODEL_ORDER)
        + "。"
    )
    for model, count in ablation_counts.items():
        if count == 1:
            lines.append(f"- {MODEL_LABELS[model]} 消融目前只找到 1 个 `ablation_results.json`，无法画重复 run 的不确定性带。")
    lines.append("- SmolLM2 目录下存在 `消融\\0` 的 PNG，但没有对应 `ablation_results.json`，本批图未纳入该 run。")
    lines.append("")
    path = OUT / "README_figures.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    for legacy_pattern in ("fig2_model_baseline_comparison.*", "fig2_layer_merge_vs_global_baselines.*"):
        for legacy in OUT.glob(legacy_pattern):
            try:
                legacy.unlink()
            except OSError:
                pass
    notes: list[str] = []

    layer_runs, layer_notes = load_layer_runs()
    notes.extend(layer_notes)
    ablation_runs, ablation_notes = load_ablation_runs()
    notes.extend(ablation_notes)
    baseline_rows, baseline_sources, baseline_notes = load_baseline_rows()
    notes.extend(baseline_notes)

    outputs: dict[str, list[Path]] = {}
    outputs["fig1_pipeline_protocol_overview"] = fig1_pipeline_protocol_overview()
    outputs["fig2_layer_merge_vs_selected_baselines"] = fig2_layer_merge_vs_selected_baselines(baseline_rows)
    outputs["fig4_layer_coefficients_overview"] = fig4_layer_coefficients_overview(layer_runs)
    outputs["fig5_ablation_tradeoff"] = fig5_ablation_tradeoff(ablation_runs)
    paths, formal_convergence_notes = figA4_formal_layer_convergence()
    outputs["figA4_formal_layer_convergence"] = paths
    notes.extend(formal_convergence_notes)
    paths, test_convergence_notes = figA4_test_reproducibility_convergence()
    outputs["figA4_test_reproducibility_convergence"] = paths
    notes.extend(test_convergence_notes)
    outputs["figA5_layer_segment_stats"] = figA5_layer_segment_stats(layer_runs)

    readme = write_readme(outputs, baseline_sources, layer_runs, ablation_runs, notes)
    print("Generated files:")
    for stem, paths in outputs.items():
        print(stem)
        for path in paths:
            print("  ", path)
    print("README", readme)


if __name__ == "__main__":
    main()
