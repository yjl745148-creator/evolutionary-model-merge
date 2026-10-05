"""Plot Figure 5 from the repository's 8B search and evaluation records."""
import math
import statistics
import yaml
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from figure_common import DATA_DIR, output_dir, setup_font, require_file, read_json, metrics
from figure05_plot_helpers import fig_search_compare

def coefficient(mode):
    path = DATA_DIR / "llama8b" / ("global_search/merge/results.yaml" if mode == "global" else "layer_search/layer_merge/results.yaml")
    record = yaml.safe_load(require_file(path).read_text(encoding="utf-8"))
    values = record["optimization_history"]["global_best_solution"]
    expected = 1 if mode == "global" else 32
    if len(values) != expected:
        raise ValueError(f"Expected {expected} coefficient logits in {path}, got {len(values)}")
    return statistics.fmean(1 / (1 + math.exp(-x)) for x in values) * 100

def main():
    setup_font()
    plt.rcParams.update({"axes.linewidth": .9, "savefig.dpi": 300, "svg.hashsalt": "emma-figure05"})
    out = output_dir()
    global_report = read_json(DATA_DIR / "llama8b/global_search/compare/llama3_8b_2_report.json")["full_results"]
    layer_report = read_json(DATA_DIR / "llama8b/layer_search/compare_layer_merge/no_system_prompt/comparison_report.json")["full_results"]
    chat = metrics(global_report["chat"])
    g, p = metrics(global_report["merged"]), metrics(layer_report["merged"])
    # The paper uses the same global-report chat reference for both DQR values.
    for values in (g, p):
        values["dqr"] = round(values["dialogue"] / chat["dialogue"] * 100, 1)
        values["asr"] = round(values["asr"], 1)
    def save(fig, stem):
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        width, height = max(2397/300, tight.width), max(1101/300, tight.height)
        box = Bbox.from_bounds(tight.x0-(width-tight.width)/2, tight.y0-(height-tight.height)/2, width, height)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"figure05_search_comparison.{ext}", bbox_inches=box, facecolor="white")
        plt.close(fig)
    fig_search_compare(coefficient("global"), coefficient("layer"), g, p, round(chat["asr"], 1), save)
    print("Figure 5 saved to", out)

if __name__ == "__main__":
    main()
