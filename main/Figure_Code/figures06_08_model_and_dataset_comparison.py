"""Figures 6 and 8 from the formal reports bundled in date/."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure_common import output_dir, setup_font, model_result, metrics, asr_counts
OUT = output_dir()
BLUE = "#2F6FB6"
TEAL = "#16836E"
RED = "#D64A3A"
NAVY = "#203F60"
INK = "#23313F"
MUTED = "#657483"
GRID = "#DCE4EA"
PALE_BLUE = "#F3F7FB"
PALE_TEAL = "#F2F8F6"



def setup_matplotlib():
    setup_font()
    plt.rcParams.update({"font.size":8.6,"axes.titlesize":10,"axes.titleweight":"bold","axes.labelsize":8.8,"xtick.labelsize":8,"ytick.labelsize":8.4,"legend.fontsize":8.3})

def save_matplotlib_figure(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + ext), dpi=300, facecolor="white")
    plt.close(fig)
    return {}

def load_figure6_data():
    return {model: {variant: metrics(model_result(model, variant)) for variant in ("merged", "chat", "base")}
            for model in ("llama3b", "llama8b", "smollm2")}

def load_table12_data():
    rows = []
    for model, label in [("llama8b", "LLaMA3-8B"), ("llama3b", "LLaMA3-3B"), ("smollm2", "SmolLM2 (1.7B)")]:
        for variant in ("merged", "chat", "base"):
            result = model_result(model, variant)
            row = {"Model": label, "Variant": variant.title()}
            for dataset in ("AdvBench", "HarmBench", "JailbreakBench", "Overall"):
                success, count = asr_counts(result, None if dataset == "Overall" else dataset)
                row[dataset] = f"{success}/{count} ({100*success/count}%)"
            rows.append(row)
    return rows

def clean_axis(ax, *, xlim: tuple[float, float], xticks: list[float], xlabel: str) -> None:
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, color=MUTED, labelpad=5)
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#AEBBC5")
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.tick_params(axis="x", colors=MUTED, length=3)

def percentage(cell: str) -> float:
    return float(cell.split("(", 1)[1].split("%", 1)[0])

def build_figure6() -> dict[str, object]:
    data = load_figure6_data()
    models = [
        ("llama3b", "LLaMA3-3B"),
        ("llama8b", "LLaMA3-8B"),
        ("smollm2", "SmolLM2 (1.7B)"),
    ]
    metrics = [
        ("dialogue", "(a) Dialogue quality", (0, 7.2), [0, 2, 4, 6], "Dialogue score (0–10)", 2),
        ("asr", "(b) Overall ASR", (0, 75), [0, 20, 40, 60], "ASR (%)", 1),
        ("composite", "(c) Composite score", (0, 6.2), [0, 2, 4, 6], "Composite score (0–10)", 2),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), sharey=True, facecolor="white")
    fig.subplots_adjust(left=0.175, right=0.985, bottom=0.16, top=0.96, wspace=0.16)
    y_values = [2.2, 1.1, 0.0]
    for ax_index, (ax, metric_spec) in enumerate(zip(axes, metrics)):
        metric, title, xlim, xticks, xlabel, decimals = metric_spec
        ax.set_facecolor(PALE_BLUE if ax_index != 1 else PALE_TEAL)
        for y in y_values:
            ax.axhline(y, color="white", linewidth=13, zorder=0)
        for (model_key, _), y in zip(models, y_values):
            merged = data[model_key]["merged"][metric]
            chat = data[model_key]["chat"][metric]
            base = data[model_key]["base"][metric]
            ax.plot([chat, merged], [y, y], color="#9BC7BC", linewidth=2.2, zorder=1)
            ax.scatter([merged], [y], s=47, color=BLUE, edgecolor="white", linewidth=0.7, zorder=4)
            ax.scatter([chat], [y], s=47, facecolor="white", edgecolor=TEAL, linewidth=1.8, zorder=4)
            base_y = y - 0.50
            ax.scatter([base], [base_y], s=39, color=RED, marker="D", edgecolor="white", linewidth=0.6, zorder=4)
            fmt = f"{{:.{decimals}f}}"
            ax.annotate(fmt.format(merged), (merged, y), xytext=(0, 8), textcoords="offset points", ha="center", color=BLUE, fontsize=7.4, weight="bold")
            ax.annotate(fmt.format(chat), (chat, y), xytext=(0, -10), textcoords="offset points", ha="center", color=TEAL, fontsize=7.2)
            ax.annotate(fmt.format(base), (base, base_y), xytext=(0, -10), textcoords="offset points", ha="center", color=RED, fontsize=7.1)
        clean_axis(ax, xlim=xlim, xticks=xticks, xlabel=xlabel)
    axes[0].set_yticks(y_values, [name for _, name in models])
    axes[0].set_ylim(-1.3, 2.65)
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=7.5, label="Merged"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=TEAL, markeredgewidth=1.6, markersize=7.5, label="Chat"),
        Line2D([], [], marker="D", color="none", markerfacecolor=RED, markeredgecolor="white", markersize=6.8, label="Base"),
    ]
    axes[0].legend(handles=handles, loc="lower right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 0.01), borderaxespad=0.1, handlelength=0.9, handletextpad=0.25, columnspacing=0.55, markerscale=0.8, fontsize=7.1)
    result = save_matplotlib_figure(
        fig,
        OUT / "figure06_model_comparison",
    )
    result["design"] = "three-panel horizontal dot comparison with paired chat-to-merged connectors"
    return result

def build_figure8() -> dict[str, object]:
    rows = load_table12_data()
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["Model"], {})[row["Variant"]] = row
    models = [
        ("LLaMA3-8B", "+18.7 pp"),
        ("LLaMA3-3B", "+48.0 pp"),
        ("SmolLM2 (1.7B)", "+28.7 pp"),
    ]
    datasets = ["AdvBench", "HarmBench", "JailbreakBench", "Overall"]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.05), sharey=True, facecolor="white")
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.19, top=0.92, wspace=0.12)
    y_values = [3.3, 2.2, 1.1, 0.0]
    for index, (ax, (model, gain)) in enumerate(zip(axes, models)):
        gain = f"{percentage(grouped[model]['Merged']['Overall']) - percentage(grouped[model]['Chat']['Overall']):+.1f} pp"
        ax.set_facecolor(PALE_BLUE if index != 1 else PALE_TEAL)
        ax.axhspan(-0.42, 0.42, color="#E8F0F7", zorder=0)
        for dataset, y in zip(datasets, y_values):
            merged = percentage(grouped[model]["Merged"][dataset])
            chat = percentage(grouped[model]["Chat"][dataset])
            base = percentage(grouped[model]["Base"][dataset])
            ax.plot([chat, merged], [y, y], color="#8FC4B8", linewidth=2.5, zorder=1)
            ax.scatter([merged], [y], s=50, color=BLUE, edgecolor="white", linewidth=0.7, zorder=4)
            ax.scatter([chat], [y], s=50, facecolor="white", edgecolor=TEAL, linewidth=1.8, zorder=4)
            base_y = y - 0.50
            ax.scatter([base], [base_y], s=41, color=RED, marker="D", edgecolor="white", linewidth=0.6, zorder=4)
            ax.annotate(f"{merged:.0f}", (merged, y), xytext=(0, 8), textcoords="offset points", ha="center", color=BLUE, fontsize=7.3, weight="bold")
            ax.annotate(f"{chat:.0f}", (chat, y), xytext=(0, -8), textcoords="offset points", ha="center", color=TEAL, fontsize=7.1)
            ax.annotate(f"{base:.0f}", (base, base_y), xytext=(0, -9), textcoords="offset points", ha="center", color=RED, fontsize=7.0)
        ax.text(0.5, -0.19, f"({chr(97 + index)}) {model}", transform=ax.transAxes, ha="center", va="top", color="black", fontsize=8.5, weight="normal")
        ax.text(0.5, 1.015, f"Overall gain vs. Chat: {gain}", transform=ax.transAxes, ha="center", va="bottom", color=TEAL, fontsize=7.5, weight="bold")
        clean_axis(ax, xlim=(0, 80), xticks=[0, 20, 40, 60, 80], xlabel="ASR (%)")
    axes[0].set_yticks(y_values, datasets)
    axes[0].set_ylim(-1.3, 3.74)
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=7.5, label="Merged"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=TEAL, markeredgewidth=1.6, markersize=7.5, label="Chat"),
        Line2D([], [], marker="D", color="none", markerfacecolor=RED, markeredgecolor="white", markersize=6.8, label="Base"),
    ]
    axes[0].legend(handles=handles, loc="lower right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 0.01), borderaxespad=0.1, handlelength=0.9, handletextpad=0.25, columnspacing=0.55, markerscale=0.8, fontsize=7.1)
    result = save_matplotlib_figure(
        fig,
        OUT / "figure08_dataset_asr",
    )
    result["design"] = "three-panel paired ASR dumbbell comparison with explicit overall gains"
    return result

def main():
    setup_matplotlib()
    build_figure6()
    build_figure8()
    print("Figures 6 and 8 saved to", OUT)

if __name__ == "__main__":
    main()
