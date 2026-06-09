#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation study: merge ratio vs ASR (Attack Success Rate).
Manually sweep the base-model weight t in {0.1, 0.2, ..., 0.9}, SLERP-merge for
each t, and evaluate ASR. Used to demonstrate the "minimal-damage cost" core idea.

Usage:
  python ablation_merge_ratio.py \
    --model-a /path/to/chat_model \
    --model-b /path/to/base_model \
    --output /path/to/output_dir
"""

import argparse
import gc
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Ensure UTF-8 output on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evolutionary_model_merge import (
    DialogueEvaluator,
    ParameterSpaceMerger,
    SafetyEvaluator,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_cpu(path: str, trust_remote_code: bool = False):
    """Load a model onto CPU (float16) to avoid using GPU memory."""
    print(f"  加载模型: {path}")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model


def report_param_overlap(model_a, model_b):
    """Print the parameter overlap ratio to avoid silent mismatches."""
    keys_a = {n for n, _ in model_a.named_parameters()}
    keys_b = {n for n, _ in model_b.named_parameters()}
    inter = keys_a & keys_b
    print(
        f"  参数重叠: {len(inter)}/{len(keys_a)} (A), "
        f"{len(inter)}/{len(keys_b)} (B)"
    )
    only_a = len(keys_a - keys_b)
    only_b = len(keys_b - keys_a)
    if only_a or only_b:
        print(f"  [WARN] 参数不完全一致: only_in_A={only_a}, only_in_B={only_b}")


def slerp_merge_models(model_a, model_b, t: float):
    """
    SLERP-merge two models on CPU and return a new (CPU) model.
    t = weight of the base model (model_b): t=0 -> pure chat, t=1 -> pure base.
    """
    merger = ParameterSpaceMerger()
    param_coeffs = np.array([1.0 - t, t], dtype=float)

    # Use model_a's structure as the template and copy its state_dict.
    model_a.cpu()
    merged = deepcopy(model_a)
    merged.cpu()

    pd_a = {name: p.data for name, p in model_a.named_parameters()}
    pd_b = {name: p.data for name, p in model_b.named_parameters()}

    merged_count = 0
    for name, param in merged.named_parameters():
        if name not in pd_a or name not in pd_b:
            continue
        pa, pb = pd_a[name], pd_b[name]
        try:
            merged_param = merger.slerp_merge([pa, pb], param_coeffs.tolist())
            param.data = merged_param.to(param.dtype)
        except Exception:
            # fallback: linear interpolation
            param.data = ((1 - t) * pa + t * pb).to(param.dtype)
        merged_count += 1

    print(f"  融合参数数: {merged_count}")
    return merged


def move_to_gpu(model):
    """Move a model to GPU (prefer bfloat16, otherwise float16)."""
    if torch.cuda.is_bf16_supported():
        return model.to(device="cuda", dtype=torch.bfloat16)
    return model.to(device="cuda", dtype=torch.float16)


def cleanup(model):
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def resolve_api_provider_and_key(args):
    provider = (args.api_provider or os.environ.get("LLM_PROVIDER", "qwen")).lower()
    if provider not in {"qwen", "openai"}:
        provider = "qwen"
    env_key = "OPENAI_API_KEY" if provider == "openai" else "QWEN_API_KEY"
    return provider, (args.api_key or os.environ.get(env_key)), env_key

def run_ablation(args):
    set_seed(args.seed)
    run_started = datetime.now()
    run_id = args.run_id or run_started.strftime("run_%Y%m%d_%H%M%S")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = args.ratios if args.ratios is not None else [
        round(t, 2) for t in np.arange(0.1, 1.0, 0.1)
    ]

    print(f"扫描比例: {ratios}")
    print(f"输出目录: {output_dir}")
    print()

    # --- Load tokenizer ---
    print("[1/4] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_a,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load both source models onto CPU ---
    print("[2/4] 加载源模型到 CPU ...")
    model_a = load_model_cpu(args.model_a, args.trust_remote_code)  # chat/instruct
    model_b = load_model_cpu(args.model_b, args.trust_remote_code)  # base

    report_param_overlap(model_a, model_b)

    # --- Initialize evaluators ---
    api_provider, api_key, api_env_key = resolve_api_provider_and_key(args)
    print(f"Safety evaluator provider: {api_provider} (env fallback: {api_env_key})")
    safety_eval = SafetyEvaluator(api_key=api_key, api_provider=api_provider)
    dialogue_eval = DialogueEvaluator()

    # --- Sweep ---
    print(f"\n[3/4] 开始扫描 {len(ratios)} 个比例点 ...\n")
    results = []
    partial_path = output_dir / "ablation_results.partial.json"

    for t in ratios:
        print(f"{'=' * 60}")
        print(f"  比例 t={t:.2f}  (chat {1-t:.2f} + base {t:.2f})")
        print(f"{'=' * 60}")

        # Merge (on CPU)
        print("  SLERP 融合中 ...")
        merged_cpu = slerp_merge_models(model_a, model_b, t)

        # Move to GPU
        print("  移到 GPU ...")
        merged_gpu = None
        try:
            merged_gpu = move_to_gpu(merged_cpu)
        except RuntimeError as e:
            print(f"  [ERR] GPU 加载失败: {e}")
            cleanup(merged_cpu)
            results.append({"t": t, "asr": None, "dialogue": None, "error": str(e)})
            continue
        cleanup(merged_cpu)  # release the CPU copy after a successful GPU move

        # Evaluate
        print("  评估中 ...")
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            safety_score = float(
                safety_eval.evaluate(
                    merged_gpu,
                    tokenizer,
                    max_new_tokens=args.gen_tokens,
                    fast_n=args.safety_fast_n,
                )
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            dialogue_score = float(
                dialogue_eval.evaluate(
                    merged_gpu,
                    tokenizer,
                    max_new_tokens=args.gen_tokens,
                    fast_n=args.dialogue_fast_n,
                )
            )
            asr = round(1.0 - safety_score, 4)
            dialogue_score = round(dialogue_score, 4)
            peak_mem_gb = None
            if torch.cuda.is_available():
                peak_mem_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        except Exception as e:
            print(f"  [ERR] 评估失败: {e}")
            cleanup(merged_gpu)
            results.append({"t": t, "asr": None, "dialogue": None, "error": str(e)})
            continue

        cleanup(merged_gpu)

        entry = {"t": t, "asr": asr, "dialogue": dialogue_score}
        if peak_mem_gb is not None:
            entry["peak_mem_gb"] = peak_mem_gb
        results.append(entry)

        mem_info = f"  峰值显存={peak_mem_gb}GB" if peak_mem_gb is not None else ""
        print(f"  结果: ASR={asr:.4f}  对话={dialogue_score:.4f}{mem_info}")

        # Write the partial file in real time
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # --- Save final results ---
    print("\n[4/4] 保存结果 ...")
    final = {
        "experiment_time": run_started.isoformat(),
        "run_id": run_id,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "ratios": ratios,
        "seed": args.seed,
        "evaluator_config": {
            "api_provider": api_provider,
            "api_key_env": api_env_key,
            "api_key_provided": bool(api_key),
            "safety_fast_n": args.safety_fast_n,
            "dialogue_fast_n": args.dialogue_fast_n,
            "gen_tokens": args.gen_tokens,
            "trust_remote_code": args.trust_remote_code,
            "plot_enabled": not args.no_plot,
        },
        "results": results,
    }
    result_path = output_dir / "ablation_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"  已保存: {result_path}")

    # Delete the partial file
    if partial_path.exists():
        partial_path.unlink()

    # --- Plot ---
    if args.no_plot:
        print("  Plot skipped (--no-plot)")
    else:
        try:
            plot_results(results, output_dir)
        except Exception as e:
            print(f"  [WARN] 绘图失败: {e}（结果已保存，可手动绘图）")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)
    print(f"  {'t':>5}  {'ASR':>8}  {'对话':>8}  {'peakGB':>8}")
    print(f"  {'-' * 38}")
    for r in results:
        asr_str = f"{r['asr']:.4f}" if r["asr"] is not None else "N/A"
        dlg_str = f"{r['dialogue']:.4f}" if r["dialogue"] is not None else "N/A"
        mem_str = f"{r['peak_mem_gb']:.3f}" if "peak_mem_gb" in r else "-"
        print(f"  {r['t']:>5.2f}  {asr_str:>8}  {dlg_str:>8}  {mem_str:>8}")


def plot_results(results, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use English labels to avoid missing CJK font issues on Linux servers
    valid = [r for r in results if r["asr"] is not None]
    if not valid:
        print("  No valid results, skipping plot")
        return

    ts = [r["t"] for r in valid]
    asrs = [r["asr"] for r in valid]
    dialogues = [r["dialogue"] for r in valid]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(ts, asrs, "r-o", linewidth=2, markersize=6, label="ASR (Harmful Rate)")
    ax1.set_xlabel("Base Model Weight t", fontsize=12)
    ax1.set_ylabel("ASR", color="r", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="r")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks([round(t, 1) for t in np.arange(0, 1.1, 0.1)])
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ts, dialogues, "b--s", linewidth=2, markersize=6, label="Dialogue Quality")
    ax2.set_ylabel("Dialogue Quality", color="b", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="b")
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Merge Ratio vs ASR / Dialogue Quality", fontsize=13)
    fig.tight_layout()

    plot_path = output_dir / "ablation_ratio_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ablation study: merge ratio vs ASR")
    p.add_argument("--model-a", required=True, help="Path to the chat/instruct model")
    p.add_argument("--model-b", required=True, help="Path to the base model")
    p.add_argument(
        "--output",
        default="experiments/ablation_ratio",
        help="Output directory (default: experiments/ablation_ratio)",
    )
    p.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=None,
        help="Custom ratio list, e.g. --ratios 0.1 0.3 0.5 0.7 0.9",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    p.add_argument("--api-provider", default=None, help="Safety evaluator API provider (default: LLM_PROVIDER or qwen)")
    p.add_argument("--api-key", default=None, help="Safety evaluator API key (default: QWEN_API_KEY or OPENAI_API_KEY by provider)")
    p.add_argument("--safety-fast-n", type=int, default=10, help="Safety fast evaluation sample count")
    p.add_argument("--dialogue-fast-n", type=int, default=10, help="Dialogue fast evaluation sample count")
    p.add_argument("--gen-tokens", type=int, default=128, help="Max new tokens per evaluation prompt")
    p.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for model/tokenizer loading")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting ablation_ratio_curve.png")
    p.add_argument("--run-id", default=None, help="Run identifier recorded in ablation_results.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation(args)
