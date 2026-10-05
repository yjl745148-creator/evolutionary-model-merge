from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import cairosvg
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont
from reportlab import rl_config
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib.colors import white


rl_config.invariant = 1

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REVIEW_V1 = ROOT / "figure_revision_preview" / "v40_comprehensive_revision_review_20260902"
OUT = ROOT / "figure_revision_preview" / "v40_round2_review_20260903"
FONT_DIR = ROOT / "Times new Roman"
FIGURE6_CSV = REVIEW_V1 / "02_figure6" / "Figure06_data_audit.csv"
TABLE12_CSV = REVIEW_V1 / "04_figure8_table12" / "Table12_dataset_asr_merged_review_v1.csv"
V40_DOCX = ROOT / "archive/论文历史版本/中文/20260917_v38_v41" / "论文_中文版_v40_修订_综合结构图表与排版优化.docx"
V40_LOCKED_SHA256 = "03FF092875233320EB8516AD2D9EF11D1941F40F5D4E3B5EA7ECDBA46CBA38B2"

BLUE = "#2F6FB6"
TEAL = "#16836E"
RED = "#D64A3A"
NAVY = "#203F60"
INK = "#23313F"
MUTED = "#657483"
GRID = "#DCE4EA"
PALE_BLUE = "#F3F7FB"
PALE_TEAL = "#F2F8F6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def strings(node):
    if isinstance(node, String):
        yield node
    for child in getattr(node, "contents", []) or []:
        yield from strings(child)


def enlarge_typography(drawing: Drawing) -> None:
    """Increase readable labels modestly while preserving math hierarchy."""
    for item in strings(drawing):
        current = float(item.fontSize)
        if current < 5.0:
            item.fontSize = current * 1.06
        elif current < 6.0:
            item.fontSize = current + 0.45
        elif current < 8.6:
            item.fontSize = current * 1.10
        else:
            item.fontSize = current + 0.55


def revise_string(
    drawing: Drawing,
    old_text: str,
    new_text: str,
    *,
    font_size: float | None = None,
) -> int:
    """Apply a targeted label revision after the global type-size pass."""
    count = 0
    for item in strings(drawing):
        if item.text != old_text:
            continue
        item.text = new_text
        if font_size is not None:
            item.fontSize = font_size
        count += 1
    return count


def remove_top_level_items(
    drawing: Drawing,
    *,
    text_values: set[str],
    rect_y_at_or_above: float | None = None,
) -> None:
    kept = []
    for item in drawing.contents:
        if isinstance(item, String) and item.text in text_values:
            continue
        if (
            rect_y_at_or_above is not None
            and isinstance(item, Rect)
            and float(item.y) >= rect_y_at_or_above
        ):
            continue
        kept.append(item)
    drawing.contents[:] = kept


def configure_round2_output(module, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    module.OUT = out_dir
    if hasattr(module, "QA_DIR"):
        module.QA_DIR = out_dir / "00_pdf_render_qa"
        module.QA_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(module, "v16"):
        module.v16.OUT = out_dir
        if hasattr(module.v16, "v14"):
            module.v16.v14.OUT = out_dir
        if hasattr(module.v16, "v12"):
            module.v16.v12.OUT = out_dir
    if hasattr(module, "v12"):
        module.v12.OUT = out_dir


def render_vector_candidate(
    drawing: Drawing,
    module,
    out_dir: Path,
    stem_name: str,
    *,
    png_width: int = 1950,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / stem_name
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    plain = out_dir / f"{stem_name}.audit_plain.svg"
    renderSVG.drawToFile(drawing, str(svg))
    module.v16.normalize_svg_fonts(svg)
    shutil.copyfile(svg, plain)
    module.v16.outline_svg_typography(svg)
    renderPDF.drawToFile(drawing, str(pdf), invariant=1)
    png_height = round(png_width * float(drawing.height) / float(drawing.width))
    cairosvg.svg2png(
        bytestring=svg.read_bytes(),
        write_to=str(png),
        output_width=png_width,
        output_height=png_height,
        background_color="white",
    )
    with Image.open(png) as image:
        rgb = Image.new("RGB", image.size, "white")
        if image.mode == "RGBA":
            rgb.paste(image, mask=image.getchannel("A"))
        else:
            rgb.paste(image.convert("RGB"))
        rgb.save(png, dpi=(300, 300))
    raw = svg.read_bytes()
    forbidden = (b"<text", b"<image", b"<filter", b"Arial")
    for token in forbidden:
        if token in raw:
            raise AssertionError(f"Forbidden SVG content {token!r}: {svg}")
    plain_text = plain.read_text(encoding="utf-8")
    if "Times New Roman" not in plain_text and "TimesNewRoman" not in plain_text:
        raise AssertionError(f"Times New Roman missing from source-text SVG: {plain}")
    return {
        "svg": str(svg.relative_to(ROOT)),
        "svg_sha256": sha256(svg),
        "png": str(png.relative_to(ROOT)),
        "png_sha256": sha256(png),
        "pdf": str(pdf.relative_to(ROOT)),
        "pdf_sha256": sha256(pdf),
        "canvas_pt": [drawing.width, drawing.height],
        "png_px": [png_width, png_height],
        "live_svg_text": 0,
        "svg_images": 0,
        "svg_filters": 0,
        "font_source": "Times New Roman; outlined to SVG paths",
    }


def build_figure1() -> dict[str, object]:
    module = load_module("round2_f1", "build_v40_review_figure1_dense.py")
    out_dir = OUT / "01_figure1"
    module.setup_typography()
    configure_round2_output(module, out_dir)
    drawing = module.figure1_dense()
    remove_top_level_items(
        drawing,
        text_values={
            "EMMA search loop: layer-wise SLERP, proxy ranking, and CMA-ES update",
            "g: generation | k=1:λ candidates | ℓ=1:L aligned layers | μ: elites",
        },
    )
    drawing.height = 460
    drawing.contents.insert(
        0,
        Rect(0, 0, drawing.width, drawing.height, fillColor=white, strokeColor=white, strokeWidth=0),
    )
    enlarge_typography(drawing)
    if revise_string(drawing, "higher = weaker safety", "higher → less safe") != 1:
        raise AssertionError("Figure 1 safety proxy label not found")
    if revise_string(drawing, "higher = better quality", "higher → better Dlg") != 1:
        raise AssertionError("Figure 1 dialogue proxy label not found")
    if revise_string(
        drawing,
        "each layer: attention | MLP | normalization tensors",
        "each layer: attention | MLP | normalization tensors",
        font_size=6.0,
    ) != 1:
        raise AssertionError("Figure 1 layer detail label not found")
    result = render_vector_candidate(
        drawing,
        module,
        out_dir,
        "Figure01_search_loop_no_header_review_v3",
    )
    result["removed_header_lines"] = 2
    return result


def build_figure2() -> dict[str, object]:
    module = load_module("round2_f2", "build_v40_review_figure2_dense.py")
    out_dir = OUT / "02_figure2"
    module.setup_typography()
    configure_round2_output(module, out_dir)
    drawing = module.figure2_dense()
    remove_top_level_items(
        drawing,
        text_values={
            "Formal evaluation: frozen models, locked protocol, traceable evidence",
            "One record per model–dataset–metric tuple; search proxies never enter the claim package.",
        },
        rect_y_at_or_above=470,
    )
    drawing.height = 456
    enlarge_typography(drawing)
    result = render_vector_candidate(
        drawing,
        module,
        out_dir,
        "Figure02_locked_evidence_no_header_review_v3",
    )
    result["removed_header_lines"] = 2
    return result


def build_figure3() -> dict[str, object]:
    module = load_module("round2_f3", "build_v40_review_figure3_dense.py")
    out_dir = OUT / "03_figure3"
    module.setup_typography()
    configure_round2_output(module, out_dir)
    records = module.parse_candidate_log()
    results, selected, layer_alphas = module.load_result_state(records)
    frontier = module.pareto_frontier(records)
    drawing = module.build_figure(
        records,
        frontier,
        selected,
        float(results["optimization_history"]["global_best_score"]),
        layer_alphas,
    )
    remove_top_level_items(
        drawing,
        text_values={
            "CMA-ES search landscape, observed proxy frontier, and state updates",
            "Observed LLaMA3-8B candidates are separated from schematic state geometry and formal post-freeze evidence.",
        },
        rect_y_at_or_above=400,
    )
    drawing.height = 390
    enlarge_typography(drawing)
    if revise_string(
        drawing,
        "SLERP per layer → selected checkpoint θ*",
        "Layer-wise SLERP → checkpoint θ*",
        font_size=5.8,
    ) != 1:
        raise AssertionError("Figure 3 checkpoint label not found")
    rank_labels = 0
    for item in strings(drawing):
        if item.text in {"1", "2", "3", "4", "5"} and math.isclose(float(item.y), 115.0):
            item.y = 106
            rank_labels += 1
    if rank_labels != 5:
        raise AssertionError(f"Figure 3 rank labels found: {rank_labels}")
    result = render_vector_candidate(
        drawing,
        module,
        out_dir,
        "Figure03_search_landscape_no_header_review_v3",
    )
    result["removed_header_lines"] = 2
    result["candidate_records"] = len(records)
    result["selected_eval"] = selected.eval
    result["selected_score"] = float(results["optimization_history"]["global_best_score"])
    return result


def setup_matplotlib() -> None:
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        path = FONT_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8.6,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 8.3,
            "svg.fonttype": "path",
            "svg.hashsalt": "emma-v40-round2-review-20260903",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_matplotlib_figure(fig, stem: Path) -> dict[str, object]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white", metadata={"Date": None})
    fig.savefig(pdf, facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    raw = svg.read_bytes()
    for token in (b"<text", b"<image", b"<filter", b"Arial"):
        if token in raw:
            raise AssertionError(f"Forbidden SVG content {token!r}: {svg}")
    with Image.open(png) as image:
        size = list(image.size)
        image.save(png, dpi=(300, 300))
    return {
        "svg": str(svg.relative_to(ROOT)),
        "svg_sha256": sha256(svg),
        "png": str(png.relative_to(ROOT)),
        "png_sha256": sha256(png),
        "pdf": str(pdf.relative_to(ROOT)),
        "pdf_sha256": sha256(pdf),
        "png_px": size,
        "live_svg_text": 0,
        "font_source": "Times New Roman; outlined to SVG paths",
    }


def load_figure6_data() -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    with FIGURE6_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            result.setdefault(row["model"], {})[row["variant"]] = {
                "dialogue": float(row["dialogue"]),
                "asr": float(row["asr_percent"]),
                "composite": float(row["composite"]),
            }
    expected = {"llama3b", "llama8b", "smollm2"}
    if set(result) != expected:
        raise AssertionError(set(result))
    return result


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


def build_figure6() -> dict[str, object]:
    data = load_figure6_data()
    models = [
        ("llama3b", "LLaMA3-3B"),
        ("llama8b", "LLaMA3-8B"),
        ("smollm2", "SmolLM2 (1.7B)"),
    ]
    metrics = [
        ("dialogue", "(a) Dialogue quality", (0, 7.2), [0, 2, 4, 6], "Score (0–10)", 2),
        ("asr", "(b) Overall ASR", (0, 75), [0, 20, 40, 60], "Attack success rate (%)", 1),
        ("composite", "(c) Composite score", (0, 6.2), [0, 2, 4, 6], "Score (0–10)", 2),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.72), sharey=True, facecolor="white")
    fig.subplots_adjust(left=0.175, right=0.985, bottom=0.20, top=0.79, wspace=0.16)
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
        ax.set_title(title, color=NAVY, pad=9)
        clean_axis(ax, xlim=xlim, xticks=xticks, xlabel=xlabel)
    axes[0].set_yticks(y_values, [name for _, name in models])
    axes[0].set_ylim(-0.92, 2.65)
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=7.5, label="Merged"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=TEAL, markeredgewidth=1.6, markersize=7.5, label="Chat"),
        Line2D([], [], marker="D", color="none", markerfacecolor=RED, markeredgecolor="white", markersize=6.8, label="Base"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.58, 0.985), handletextpad=0.45, columnspacing=1.4)
    result = save_matplotlib_figure(
        fig,
        OUT / "04_figure6" / "Figure06_dot_comparison_review_v2",
    )
    result["design"] = "three-panel horizontal dot comparison with paired chat-to-merged connectors"
    return result


def load_table12_data() -> list[dict[str, str]]:
    with TABLE12_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 9:
        raise AssertionError(len(rows))
    return rows


def percentage(cell: str) -> float:
    return float(cell.split("(", 1)[1].split("%", 1)[0])


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
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.92), sharey=True, facecolor="white")
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.17, top=0.79, wspace=0.12)
    y_values = [3.3, 2.2, 1.1, 0.0]
    for index, (ax, (model, gain)) in enumerate(zip(axes, models)):
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
        ax.set_title(f"({chr(97 + index)}) {model}", color=NAVY, pad=15)
        ax.text(0.5, 1.015, f"Overall gain vs. Chat: {gain}", transform=ax.transAxes, ha="center", va="bottom", color=TEAL, fontsize=7.5, weight="bold")
        clean_axis(ax, xlim=(0, 80), xticks=[0, 20, 40, 60, 80], xlabel="ASR (%)")
    axes[0].set_yticks(y_values, datasets)
    axes[0].set_ylim(-0.82, 3.74)
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=7.5, label="Merged"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=TEAL, markeredgewidth=1.6, markersize=7.5, label="Chat"),
        Line2D([], [], marker="D", color="none", markerfacecolor=RED, markeredgecolor="white", markersize=6.8, label="Base"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.58, 0.985), handletextpad=0.45, columnspacing=1.4)
    result = save_matplotlib_figure(
        fig,
        OUT / "05_figure8" / "Figure08_paired_asr_review_v2",
    )
    result["design"] = "three-panel paired ASR dumbbell comparison with explicit overall gains"
    return result


def build_table12_preview() -> dict[str, object]:
    rows = load_table12_data()
    headers = ["Model", "Variant", "AdvBench", "HarmBench", "JailbreakBench", "Overall"]
    cell_rows: list[list[str]] = []
    previous_model = None
    for row in rows:
        model = row["Model"] if row["Model"] != previous_model else ""
        previous_model = row["Model"]
        cell_rows.append([model, row["Variant"], row["AdvBench"], row["HarmBench"], row["JailbreakBench"], row["Overall"]])
    fig, ax = plt.subplots(figsize=(6.5, 3.35), facecolor="white")
    ax.axis("off")
    fig.text(0.5, 0.955, "Table 12. Dataset-level ASR across Three Model Scales.", ha="center", va="top", color="#427FC2", fontsize=10.2)
    table = ax.table(
        cellText=cell_rows,
        colLabels=headers,
        colWidths=[0.185, 0.095, 0.165, 0.165, 0.205, 0.185],
        cellLoc="center",
        bbox=[0.02, 0.06, 0.96, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#C7D1D9")
        cell.set_linewidth(0.45)
        cell.PAD = 0.08
        cell.get_text().set_fontfamily("Times New Roman")
        cell.get_text().set_color(INK)
        if row_index == 0:
            cell.set_facecolor("#EAF1F7")
            cell.get_text().set_weight("bold")
            cell.set_edgecolor(NAVY)
            cell.set_linewidth(0.8)
        elif row_index in (1, 2, 3):
            cell.set_facecolor("#FAFCFE")
        elif row_index in (4, 5, 6):
            cell.set_facecolor("#F5FAF8")
        else:
            cell.set_facecolor("#FCFBF8")
        if row_index in (3, 6, 9):
            cell.set_edgecolor("#718292")
            cell.set_linewidth(0.8)
        if col_index in (0, 1):
            cell.get_text().set_ha("left")
    result = save_matplotlib_figure(
        fig,
        OUT / "06_table12" / "Table12_visible_rules_review_v2",
    )
    result["design"] = "light full grid with stronger header and model-group separators"
    return result


def tnr_font(filename: str, size: int):
    path = FONT_DIR / filename
    return ImageFont.truetype(str(path), size=size)


def make_contact_sheet(paths: list[Path], labels: list[str], output: Path) -> Path:
    width = 1500
    margin = 44
    gap = 42
    label_height = 62
    prepared: list[tuple[Image.Image, str]] = []
    total_height = margin
    for path, label in zip(paths, labels):
        image = Image.open(path).convert("RGB")
        target_width = width - 2 * margin
        target_height = round(image.height * target_width / image.width)
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        prepared.append((image, label))
        total_height += label_height + target_height + gap
    canvas = Image.new("RGB", (width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = tnr_font("timesbd.ttf", 31)
    y = margin
    for image, label in prepared:
        draw.text((margin, y), label, font=title_font, fill=NAVY)
        y += label_height
        canvas.paste(image, (margin, y))
        y += image.height + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(180, 180))
    return output


def main() -> None:
    if sha256(V40_DOCX) != V40_LOCKED_SHA256:
        raise AssertionError("v40 changed before candidate generation")
    OUT.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()
    results = {
        "figure1": build_figure1(),
        "figure2": build_figure2(),
        "figure3": build_figure3(),
        "figure6": build_figure6(),
        "figure8": build_figure8(),
        "table12": build_table12_preview(),
    }
    figure123 = make_contact_sheet(
        [ROOT / results[key]["png"] for key in ("figure1", "figure2", "figure3")],
        ["Figure 1 candidate v3", "Figure 2 candidate v3", "Figure 3 candidate v3"],
        OUT / "00_contact_sheets" / "Figure01_03_round2_candidates.png",
    )
    figure6812 = make_contact_sheet(
        [ROOT / results[key]["png"] for key in ("figure6", "figure8", "table12")],
        ["Figure 6 candidate v2", "Figure 8 candidate v2", "Table 12 line treatment"],
        OUT / "00_contact_sheets" / "Figure06_08_Table12_round2_candidates.png",
    )
    results["contact_sheets"] = {
        "figure1_3": str(figure123.relative_to(ROOT)),
        "figure1_3_sha256": sha256(figure123),
        "figure6_8_table12": str(figure6812.relative_to(ROOT)),
        "figure6_8_table12_sha256": sha256(figure6812),
    }
    if sha256(V40_DOCX) != V40_LOCKED_SHA256:
        raise AssertionError("v40 changed during candidate generation")
    payload = {
        "status": "CANDIDATES_ONLY_NOT_INTEGRATED",
        "v40_docx": str(V40_DOCX),
        "v40_sha256_unchanged": sha256(V40_DOCX),
        "results": results,
    }
    audit = OUT / "candidate_audit.json"
    audit.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PASS output={OUT}")
    print(f"PASS candidates=6 contact_sheets=2")
    print(f"PASS v40_unchanged={sha256(V40_DOCX)}")
    print(f"PASS audit={audit}")


if __name__ == "__main__":
    main()
