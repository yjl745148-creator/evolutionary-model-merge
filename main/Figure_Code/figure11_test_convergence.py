from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_and_tables" / "figures"
VECTOR_DIR = ROOT / "矢量图"
FONT_DIR = ROOT / "Times new Roman"

LATIN = "Times New Roman"
MODEL_ORDER = ["llama3b", "llama8b", "smollm2"]
MODEL_LABELS = {
    "llama3b": "Llama-3.2-3B",
    "llama8b": "Llama-3.1-8B",
    "smollm2": "SmolLM2 (1.7B)",
}
MODEL_COLORS = {
    "llama3b": "#2f5f8f",
    "llama8b": "#2f8f5b",
    "smollm2": "#8b4f9f",
}
GRAY = "#5f6670"
LIGHT_GRAY = "#d7dce2"


@dataclass
class Curve:
    model: str
    run_label: str
    source: Path
    evals: np.ndarray
    progress: np.ndarray
    score: np.ndarray
    best: np.ndarray
    achieved: np.ndarray
    final_best: float
    reach80_progress: float | None
    reach95_progress: float | None


def register_times() -> None:
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        path = FONT_DIR / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def configure_fonts() -> None:
    register_times()
    mpl.rcParams.update(
        {
            "font.family": LATIN,
            "font.serif": [LATIN],
            "mathtext.fontset": "custom",
            "mathtext.rm": LATIN,
            "mathtext.it": f"{LATIN}:italic",
            "mathtext.bf": f"{LATIN}:bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def classify_history(path: Path) -> tuple[str, str] | None:
    parts = {part.lower() for part in path.parts}
    if "llama3b" in parts:
        run = path.parent.name if path.parent.name.isdigit() else "1"
        return "llama3b", f"Run {run}"
    if "llama8b" in parts:
        return "llama8b", "Available run"
    if "smollm2" in parts:
        return "smollm2", "Available run"
    return None


def load_curve(path: Path) -> Curve | None:
    classified = classify_history(path)
    if classified is None:
        return None
    model, run_label = classified
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        return None

    evals = np.array([row.get("eval", i + 1) for i, row in enumerate(data)], dtype=float)
    fitness = np.array(
        [row.get("fitness", row.get("fitness_fast", np.nan)) for row in data],
        dtype=float,
    )
    score = -fitness
    best = np.maximum.accumulate(score)
    progress = evals / float(np.nanmax(evals)) * 100.0

    start_best = float(best[0])
    final_best = float(best[-1])
    span = final_best - start_best
    if abs(span) < 1e-12:
        achieved = np.full_like(best, 100.0)
        reach80 = None
        reach95 = None
    else:
        achieved = np.clip((best - start_best) / span * 100.0, 0.0, 100.0)
        reach80 = float(progress[np.argmax(achieved >= 80.0)]) if np.any(achieved >= 80.0) else None
        reach95 = float(progress[np.argmax(achieved >= 95.0)]) if np.any(achieved >= 95.0) else None

    return Curve(
        model=model,
        run_label=run_label,
        source=path,
        evals=evals,
        progress=progress,
        score=score,
        best=best,
        achieved=achieved,
        final_best=final_best,
        reach80_progress=reach80,
        reach95_progress=reach95,
    )


def load_curves() -> list[Curve]:
    curves = []
    for path in ROOT.rglob("history.json"):
        curve = load_curve(path)
        if curve is not None:
            curves.append(curve)
    curves.sort(key=lambda c: (MODEL_ORDER.index(c.model), c.run_label))
    return curves


def save_all(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        FIG_DIR / f"{stem}.png",
        FIG_DIR / f"{stem}.pdf",
        FIG_DIR / f"{stem}.svg",
        VECTOR_DIR / "Figure12_v14.png",
        VECTOR_DIR / "Figure12_v14.pdf",
        VECTOR_DIR / "Figure12_v14.svg",
    ]
    for path in outputs:
        if path.suffix == ".png":
            fig.savefig(path, dpi=300)
        else:
            fig.savefig(path)
    return outputs


def format_reach(curves: list[Curve], attr: str) -> str:
    values = [getattr(curve, attr) for curve in curves if getattr(curve, attr) is not None]
    if not values:
        return "not applicable"
    if len(values) == 1:
        return f"{values[0]:.0f}% progress"
    return f"{min(values):.0f}-{max(values):.0f}% progress"


def draw() -> list[Path]:
    configure_fonts()
    curves = load_curves()
    by_model = {model: [c for c in curves if c.model == model] for model in MODEL_ORDER}

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65), sharey=True)
    legend_handles = []
    legend_labels = []

    for ax, model in zip(axes, MODEL_ORDER):
        model_curves = by_model[model]
        color = MODEL_COLORS[model]
        if not model_curves:
            ax.text(0.5, 0.5, "No available history", ha="center", va="center", transform=ax.transAxes)
            continue

        grid = np.linspace(0, 100, 201)
        interpolated = []
        for curve in model_curves:
            y = np.interp(grid, curve.progress, curve.achieved)
            interpolated.append(y)
            label = "Individual run" if len(model_curves) > 1 and not legend_handles else None
            line = ax.plot(
                curve.progress,
                curve.achieved,
                color=color,
                lw=1.25 if len(model_curves) > 1 else 2.4,
                alpha=0.28 if len(model_curves) > 1 else 0.95,
                drawstyle="steps-post",
                label=label,
            )[0]
            if label:
                legend_handles.append(line)
                legend_labels.append(label)

        stack = np.vstack(interpolated)
        if len(model_curves) > 1:
            band = ax.fill_between(
                grid,
                np.nanmin(stack, axis=0),
                np.nanmax(stack, axis=0),
                color=color,
                alpha=0.12,
                linewidth=0,
                label="Run range",
            )
            mean_line = ax.plot(
                grid,
                np.nanmean(stack, axis=0),
                color=color,
                lw=2.8,
                drawstyle="steps-post",
                label="Mean trajectory",
            )[0]
            legend_handles.extend([band, mean_line])
            legend_labels.extend(["Run range", "Mean trajectory"])
        else:
            single = ax.lines[-1]
            if "Available run" not in legend_labels:
                legend_handles.append(single)
                legend_labels.append("Available run")

        threshold = ax.axhline(80, color="#8b939e", lw=0.85, ls=(0, (4, 3)), label="80% of final improvement")
        if "80% of final improvement" not in legend_labels:
            legend_handles.append(threshold)
            legend_labels.append("80% of final improvement")

        final_scores = [c.final_best for c in model_curves]
        if len(model_curves) == 1:
            score_text = f"final score = {final_scores[0]:.3f}"
        else:
            score_text = f"final score range = {min(final_scores):.3f}-{max(final_scores):.3f}"
        ax.text(
            0.03,
            0.08,
            f"80% reached: {format_reach(model_curves, 'reach80_progress')}\n"
            f"95% reached: {format_reach(model_curves, 'reach95_progress')}\n"
            f"{score_text}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.2,
            color=GRAY,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": LIGHT_GRAY, "linewidth": 0.55},
        )
        run_word = "available runs" if len(model_curves) > 1 else "available run"
        ax.set_title(f"{MODEL_LABELS[model]} ({len(model_curves)} {run_word})", loc="left", fontsize=8.0, weight="bold")
        ax.set_xlabel("Evaluation progress (%)", fontsize=7.8)
        ax.set_xlim(0, 100)
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="y", color="#d9dde3", lw=0.65, alpha=0.85)
        ax.grid(axis="x", color="#eef1f4", lw=0.55, alpha=0.70)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#aeb5be")
        ax.spines["bottom"].set_color("#aeb5be")
        ax.tick_params(labelsize=7.4, length=2.8)

    axes[0].set_ylabel("Best-so-far improvement achieved (%)", fontsize=7.8)
    fig.suptitle("Test-version per-layer search convergence", x=0.06, y=0.985, ha="left", fontsize=10.2, weight="bold")
    fig.text(
        0.06,
        0.907,
        "Best-so-far improvement toward each run's final score; full-evaluation checkpoints are omitted.",
        ha="left",
        va="top",
        fontsize=7.2,
        color=GRAY,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=7.1,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.75, bottom=0.32, wspace=0.25)
    return save_all(fig, "figA4_test_reproducibility_convergence_v14")


def main() -> None:
    outputs = draw()
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
