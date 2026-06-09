#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_merge_result.py
Read a per-layer search output directory and draw a result figure in the same
format as layer_heatmap.py. The convergence curve is parsed preferentially from
run.log (per-eval scores); without a log it falls back to per-generation scores.

Usage:
  python plot_merge_result.py \
    --output-dir experiments/llama3b_perlayer

  python plot_merge_result.py \
    --output-dir experiments/llama3b_perlayer \
    --log run.log \
    --save experiments/llama3b_perlayer/heatmap.png
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_best_t(output_dir: Path):
    """Read the optimal per-layer t values from best_so_far.json (logit -> sigmoid)."""
    p = output_dir / "best_so_far.json"
    if not p.exists():
        raise FileNotFoundError(
            f"找不到 {p}\n"
            "请确认 --output-dir 指向包含 best_so_far.json 的实验目录，\n"
            "且该实验使用了 per_layer_search=True。"
        )
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    sol = data.get("best_solution")
    if sol is None:
        raise ValueError("best_so_far.json 中没有 best_solution 字段")
    return sigmoid(sol), data


# Matches each evaluation line printed by evolutionary_model_merge (kept in
# Chinese on purpose: it must match the exact log text that script emits, e.g.
# "评估结果 - 安全: 0.450, 对话: 0.720, PPL: 45.3(得分0.85), 总分: 0.345×0.93=0.321").
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_RE_EVAL = re.compile(
    rf"评估结果\s*-\s*安全:\s*({_NUMBER}),\s*对话:\s*({_NUMBER}),"
    rf".*?总分:\s*{_NUMBER}(?:×{_NUMBER})+=\s*({_NUMBER})"
)
# Matches gate-penalty lines (triggered by the PPL / dialogue gate). Kept in
# Chinese to match the emitted log text.
_RE_GATE = re.compile(
    rf"\[GATE-(?:PPL|DLG|REP)[^\]]*\].*?判为惩罚\s*({_NUMBER})"
)


def load_eval_from_log(log_path: Path):
    """
    Parse run.log and extract, in order, each evaluation's
    (safety, dialogue, total_score). Gate-penalty lines are recorded as
    (nan, nan, penalty). Returns (evals, safeties, dialogues, scores) as lists.
    """
    if not log_path.exists():
        return [], [], [], []

    evals, safeties, dialogues, scores = [], [], [], []
    idx = 0

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _RE_EVAL.search(line)
            if m:
                idx += 1
                safeties.append(float(m.group(1)))
                dialogues.append(float(m.group(2)))
                scores.append(float(m.group(3)))
                evals.append(idx)
                continue
            mg = _RE_GATE.search(line)
            if mg:
                idx += 1
                safeties.append(float("nan"))
                dialogues.append(float("nan"))
                scores.append(float(mg.group(1)))  # penalty value
                evals.append(idx)

    return evals, safeties, dialogues, scores


def load_gen_history(output_dir: Path):
    """
    Fallback: read per-generation best/mean scores.
    Returns (generations, best_scores, mean_scores).
    """
    p = output_dir / "generation_best.jsonl"
    if p.exists():
        xs, best_scores, mean_scores = [], [], []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    best = item.get("best_score")
                    if best is not None:
                        best_scores.append(float(best))
                        mean_scores.append(float(item.get("mean_score", float("nan"))))
                        # Multi-stage search restarts "generation" from 1; plot in global order.
                        xs.append(len(xs) + 1)
                except Exception:
                    continue
        if best_scores:
            return xs, best_scores, mean_scores

    p2 = output_dir / "merged_model" / "merge_history.json"
    if p2.exists():
        with open(p2, encoding="utf-8") as f:
            hist = json.load(f)
        best_scores = [float(s) for s in hist.get("best_scores", [])]
        mean_scores = [float(s) for s in hist.get("mean_scores",
                       [float("nan")] * len(best_scores))]
        xs = list(hist.get("generations", range(1, len(best_scores) + 1)))
        if best_scores:
            return xs, best_scores, mean_scores

    return [], [], []


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(t_values: np.ndarray,
         eval_data,       # (evals, safeties, dialogues, scores) from log, or None
         gen_data,        # (xs, best_scores, mean_scores) fallback
         output_dir: Path,
         save_path: Path):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(t_values)

    use_eval = eval_data is not None and len(eval_data[0]) > 0
    use_gen  = not use_eval and gen_data is not None and len(gen_data[0]) > 0
    has_curve = use_eval or use_gen

    width_ratios = [2, 1, 2] if has_curve else [2, 1]
    ncols = 3 if has_curve else 2
    fig = plt.figure(figsize=(16 if has_curve else 10, max(5, n * 0.4)))
    gs = GridSpec(1, ncols, width_ratios=width_ratios, figure=fig)

    # -- Panel 1: heatmap --------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(t_values.reshape(-1, 1), cmap="coolwarm",
                    aspect="auto", vmin=0, vmax=1)
    ax1.set_xticks([0])
    ax1.set_xticklabels(["Base Weight t"])
    ax1.set_yticks(range(n))
    ax1.set_yticklabels([f"Layer {i}" for i in range(n)], fontsize=8)
    ax1.set_title(
        "Per-Layer Merge Coefficient\n(blue=chat dominant, red=base dominant)",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax1, label="t (base weight)")

    # -- Panel 2: per-layer t line ----------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t_values, range(n), "ko-", markersize=4, linewidth=1.5)
    ax2.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.fill_betweenx(range(n), t_values, 0.5,
                      where=(t_values > 0.5), color="red", alpha=0.15,
                      label="base dominant")
    ax2.fill_betweenx(range(n), t_values, 0.5,
                      where=(t_values < 0.5), color="blue", alpha=0.15,
                      label="chat dominant")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, n - 0.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Base Weight t")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([str(i) for i in range(n)], fontsize=7)
    ax2.set_title("t per Layer", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, loc="lower right")

    # -- Panel 3: convergence curve ---------------------------------------
    if use_eval:
        evals, safeties, dialogues, scores = eval_data
        scores_arr    = np.array(scores,    dtype=float)
        safeties_arr  = np.array(safeties,  dtype=float)
        dialogues_arr = np.array(dialogues, dtype=float)

        ax3 = fig.add_subplot(gs[2])
        # fitness = total_score (already positive in the log).
        # Gate-penalty points have scores far above the normal range; mark them separately.
        normal_mask  = scores_arr < 5.0
        penalty_mask = ~normal_mask

        ax3.plot(np.array(evals)[normal_mask], -scores_arr[normal_mask],
                 "g-", linewidth=1, alpha=0.7, label="fitness")
        if penalty_mask.any():
            ax3.scatter(np.array(evals)[penalty_mask], [0] * penalty_mask.sum(),
                        marker="x", color="gray", s=30, zorder=5, label="gate penalty")

        ax3b = ax3.twinx()
        valid = ~np.isnan(safeties_arr)
        if valid.any():
            ax3b.plot(np.array(evals)[valid], (1 - safeties_arr[valid]),
                      "r--", linewidth=1.5, label="ASR (1-safety)")
            ax3b.plot(np.array(evals)[valid], dialogues_arr[valid],
                      "b--", linewidth=1.5, label="Dialogue")
        ax3b.set_ylim(0, 1)
        ax3b.set_ylabel("ASR / Dialogue", fontsize=9)

        ax3.set_xlabel("Evaluation #  [每点 = 1次个体评估]")
        ax3.set_ylabel("Fitness", fontsize=9, color="g")
        ax3.set_title("Optimization Curve (per eval)", fontsize=10)
        ax3.grid(True, alpha=0.3)

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    elif use_gen:
        xs, best_scores, mean_scores = gen_data
        best_arr = np.array(best_scores, dtype=float)
        mean_arr = np.array(mean_scores, dtype=float)
        best_arr = -best_arr
        mean_arr = -mean_arr

        ax3 = fig.add_subplot(gs[2])
        ax3.plot(xs, best_arr, "g-o", markersize=3, linewidth=1.5, label="best (per gen)")
        valid_mean = ~np.isnan(mean_arr)
        if valid_mean.any():
            ax3.plot(np.array(xs)[valid_mean], mean_arr[valid_mean],
                     "b--", linewidth=1, alpha=0.6, label="mean (per gen)")
        ax3.set_xlabel("Generation  [每点 = 1代，含 popsize 次 eval]")
        ax3.set_ylabel("Fitness", fontsize=9)
        ax3.set_title("Optimization Curve (per generation)", fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

    # -- Title -------------------------------------------------------------
    subtitle = str(output_dir)
    results_yaml = output_dir / "results.yaml"
    if results_yaml.exists():
        try:
            import yaml
            with open(results_yaml, encoding="utf-8") as f:
                res = yaml.safe_load(f)
            a = Path(res.get("model_a", "")).name
            b = Path(res.get("model_b", "")).name
            if a and b:
                subtitle = f"{a}  ×  {b}"
        except Exception:
            pass

    fig.suptitle(
        "Layer-wise CMA-ES Optimal Merge Coefficients\n" + subtitle,
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图已保存: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Plot per-layer search results from evolutionary_model_merge"
    )
    p.add_argument("--output-dir", required=True,
                   help="Experiment output directory (contains best_so_far.json)")
    p.add_argument("--log", default="run.log",
                   help="Log file name (relative to --output-dir, default run.log)")
    p.add_argument("--save", default=None,
                   help="Save path (default: <output-dir>/layer_heatmap.png)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    log_path   = output_dir / args.log
    save_path  = Path(args.save) if args.save else output_dir / "layer_heatmap.png"

    print(f"读取目录: {output_dir}")

    t_values, meta = load_best_t(output_dir)
    print(f"  层数       : {len(t_values)}")
    print(f"  最优分数   : {meta.get('best_score', 'N/A')}")
    print(f"  来源       : {meta.get('stage', '?')} / 第 {meta.get('generation', '?')} 代")
    print(f"  t 均值     : {t_values.mean():.3f}  "
          f"min={t_values.min():.3f}  max={t_values.max():.3f}")

    # Preferred: parse each eval from the log.
    eval_data = None
    evals, safeties, dialogues, scores = load_eval_from_log(log_path)
    if evals:
        print(f"  日志解析   : {len(evals)} 次 eval（来自 {log_path.name}）")
        eval_data = (evals, safeties, dialogues, scores)
    else:
        print(f"  未找到/解析日志 {log_path.name}，尝试读每代历史")

    # Fallback: per-generation scores.
    gen_data = None
    if eval_data is None:
        xs, best_scores, mean_scores = load_gen_history(output_dir)
        if best_scores:
            print(f"  历史记录   : {len(best_scores)} 代（每代含 popsize 次 eval）")
            gen_data = (xs, best_scores, mean_scores)
        else:
            print("  未找到历史记录，跳过优化曲线（只画前两栏）")

    plot(t_values, eval_data, gen_data, output_dir, save_path)


if __name__ == "__main__":
    main()
