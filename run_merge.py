#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Evolutionary Model Merge

Paper-reproducible entry point: given any two same-architecture HuggingFace
models, it runs CMA-ES evolutionary optimization + SLERP/DoRA/DARE merging +
evaluation + saving automatically.

WARNING - model order convention (important!):
  --model-a = chat / instruct model  (i.e. model_a_role='chat')
  --model-b = base model             (i.e. model_b_role='base')
  Swapping the order makes the evolutionary optimization converge backwards, so
  keep it consistent.

------------------------------------------------------------------------
Prerequisite: an LLM-judge API key (used to score fitness during evaluation)
  Default uses Qwen (DashScope):
    export QWEN_API_KEY="sk-xxxx"             # Linux / macOS
    setx   QWEN_API_KEY "sk-xxxx"             # Windows (reopen the terminal)

  Switch to OpenAI ChatGPT:
    export LLM_PROVIDER=openai
    export OPENAI_API_KEY="sk-xxxx"
------------------------------------------------------------------------

Usage example (a single command listing every optional flag; trim/edit as needed):

  # ------------------------------------------------------------------
  # Full example: every CLI option, with the meaning of each line.
  # In practice, trim as needed; the minimum is just --model-a / --model-b.
  # Note: paired flags marked [mutually exclusive] allow only one; flags marked
  # [standalone mode] switch to a completely different run flow
  # (e.g. --resume / --per-layer-search).
  # ------------------------------------------------------------------
  python run_merge.py \
    `# ======== required: the two model paths (order matters!) ========` \
    --model-a /path/to/chat_model           `# [required] chat / instruct model path (HF local dir or hub repo id)` \
    --model-b /path/to/base_model           `# [required] base model path; swapping makes evolution converge backwards` \
    \
    `# ======== common: output / experiment name / config / force ========` \
    --output /path/to/output_dir            `# result dir; omit to auto-generate experiments/{A}_x_{B}` \
    --name "My Evolutionary Merge"          `# experiment name; omit to join the model names` \
    --config my_config.yaml                 `# optional YAML overriding default hyperparams (priority: CLI > YAML > default)` \
    --force                                 `# run anyway on architecture mismatch (default aborts)` \
    --resume                                `# [standalone mode] skip optimization; rebuild and save directly from best_so_far.json` \
    \
    `# ======== CMA-ES three-stage hyperparams (coarse -> fine -> refine) ========` \
    `#   per-generation cost = popsize x (safety_fast_n + dialogue_fast_n) inferences` \
    --popsize 10                            `# stage1 (coarse) population size (typically 8~12)` \
    --generations 12                        `# stage1 (coarse) max generations (typically 10~15)` \
    --sigma 0.12                            `# stage1 (coarse) initial step (in the [0,1] space, typically 0.10~0.15)` \
    --stage2-popsize 8                      `# stage2 (fine) population size (typically 6~8)` \
    --stage2-generations 8                  `# stage2 (fine) max generations (typically 6~8)` \
    --stage2-sigma 0.06                     `# stage2 (fine) initial step (typically 0.05~0.07)` \
    --stage3-popsize 6                      `# stage3 (refine) population size (typically 4~6)` \
    --stage3-generations 5                  `# stage3 (refine) max generations (typically 3~5)` \
    --stage3-sigma 0.02                     `# stage3 (refine) initial step (typically 0.01~0.03)` \
    `# --no-two-stage-cma`                  `# add this to run only stage1 (good for a smoke test; mutually exclusive with stage2/3)` \
    \
    `# ======== per-dimension CMA sigma multipliers (advanced, default follows stage_sigma) ========` \
    --cma-sigma-param   1.0                 `# sigma multiplier for the SLERP-coefficient dims (>1 = larger steps)` \
    --cma-sigma-routing 1.0                 `# sigma multiplier for the routing dims` \
    --cma-sigma-dare    1.0                 `# sigma multiplier for the DARE drop_rate dim` \
    \
    `# ======== early stop (stage3 only) ======== [mutually exclusive with disable]` \
    --enable-early-stop                     `# enable early stop; use --disable-early-stop to turn it off explicitly` \
    --early-stop-patience 3                 `# stop after this many generations with no significant improvement (typically 3)` \
    --early-stop-min-delta 0.003            `# minimum improvement threshold; below this counts as "no improvement" (typically 0.003)` \
    \
    `# ======== evaluation config (per-individual question count / generation length in evolution) ========` \
    --safety-fast-n 20                      `# safety evaluation question count (more = steadier but slower)` \
    --dialogue-fast-n 15                    `# dialogue evaluation question count` \
    --gen-tokens-fast 96                    `# max generated tokens during eval (shorter = much faster)` \
    --gate-penalty 10.0                     `# fixed penalty when the PPL/dialogue gate fails (avoids +inf breaking CMA covariance)` \
    \
    `# ======== Safety API fuzzy band (call the LLM judge only when rule_score falls in this band) ========` \
    --safety-api-judge-low  0.30            `# lower bound: rule_score < 0.30 is judged safe without an API call` \
    --safety-api-judge-high 0.75            `# upper bound: rule_score > 0.75 is judged harmful without an API call` \
    \
    `# ======== hardware / large-model support ========` \
    --large-model never                     `# large-model sharded loading: auto(>4B auto) / always(force) / never(off)` \
    --merge-device-mode gpu_full           `# merge location: gpu_full(fast but 2x VRAM) / gpu_param(saves VRAM, recommended)` \
    --trust-remote-code                     `# allow running custom modeling_*.py from the model repo (trusted sources only, else RCE risk)` \
    \
    `# ======== per-layer search [standalone mode] (independent t per layer, analyze safety-alignment distribution) ========` \
    `# Note: with --per-layer-search you must use exactly 2 models; --large-model must be never` \
    --per-layer-search                      `# enable per-layer search (off by default)` \
    --per-layer-alpha 0.7                   `# score = alpha*risk + (1-alpha)*dialogue (CMA minimizes its negative)`

  # ------------------------------------------------------------------
  # Tip: the command above lists every option; in practice trim by scenario:
  #   - Minimal quick check     : keep only --model-a / --model-b
  #   - Formal paper experiment : drop the --resume / --per-layer-search blocks
  #   - Per-layer analysis      : drop the --no-two-stage-cma block; keep --per-layer-search
  #   - Resume / re-save        : keep required + --output + --resume
  #   - Centralize hyperparams  : keep required + --output + --config <yaml>
  # ------------------------------------------------------------------

Output directory layout:
  {output}/
    +- merged_model/          merged model + tokenizer (HuggingFace format)
    +- results.yaml           experiment config + optimization history + best score
    +- generation_best.jsonl  best individual per generation (JSONL, for curves)
    +- best_so_far.json       current global-best parameters (used by --resume)
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Ensure Windows console can print UTF-8 safely
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Add the project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evolutionary_model_merge import EvolutionaryModelMerger, MergeConfig

# ---------------------------------------------------------------------------
# Architecture compatibility check
# ---------------------------------------------------------------------------

def _is_same_or_child_path(path: str, parent: str) -> bool:
    try:
        child_abs = os.path.normcase(os.path.abspath(path))
        parent_abs = os.path.normcase(os.path.abspath(parent))
        return child_abs == parent_abs or os.path.commonpath([child_abs, parent_abs]) == parent_abs
    except (OSError, ValueError):
        return False


def validate_output_not_inside_source(args):
    """Avoid writing experiment output into a source-model dir, which would pollute its checkpoints/config/tokenizer files."""
    output = str(Path(args.output))
    conflicts = []
    for label, model_path in (("model_a", args.model_a), ("model_b", args.model_b)):
        if os.path.isdir(model_path) and _is_same_or_child_path(output, model_path):
            conflicts.append(f"{label}={model_path}")

    if conflicts:
        joined = "; ".join(conflicts)
        raise ValueError(
            "--output 不能设置为源模型目录或其子目录，否则会污染源模型文件。"
            f" 当前 output={args.output}, 冲突: {joined}"
        )


def check_architecture_compatibility(model_a: str, model_b: str) -> bool:
    """
    Check that the two models' key structural fields match.
    Reads a local config.json directly if present, otherwise tries AutoConfig.
    """
    from transformers import AutoConfig

    def _load_architecture(path_or_name: str) -> dict:
        # Prefer a local config.json (fast and offline)
        local_cfg = os.path.join(path_or_name, "config.json")
        if os.path.isfile(local_cfg):
            with open(local_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg
        # Fall back to AutoConfig
        try:
            cfg = AutoConfig.from_pretrained(path_or_name, trust_remote_code=False)
            return cfg.to_dict()
        except Exception:
            return {}

    cfg_a = _load_architecture(model_a)
    cfg_b = _load_architecture(model_b)
    type_a = cfg_a.get("model_type", "unknown")
    type_b = cfg_b.get("model_type", "unknown")

    print(f"  Model A 架构: {type_a}  ({model_a})")
    print(f"  Model B 架构: {type_b}  ({model_b})")

    if type_a == "unknown" or type_b == "unknown":
        print("[WARN] 无法确定其中一个模型的架构，将继续尝试融合")
        return True

    if type_a != type_b:
        print(f"[ERR] 架构不匹配！{type_a} != {type_b}")
        print("      进化融合要求两个模型完全同架构（如 base + instruct 同系列）")
        return False

    fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "tie_word_embeddings",
    )
    mismatches = []
    for field in fields:
        value_a = cfg_a.get(field)
        value_b = cfg_b.get(field)
        if value_a is not None and value_b is not None and value_a != value_b:
            mismatches.append(f"{field}: {value_a} != {value_b}")
    if mismatches:
        print("[ERR] 模型关键结构不匹配：")
        for mismatch in mismatches:
            print(f"      - {mismatch}")
        return False

    print(f"[OK] 架构一致: {type_a}")
    return True


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def build_config(args) -> MergeConfig:
    """Build a MergeConfig from CLI args (optionally layering in YAML)."""

    # Base defaults
    defaults = dict(
        param_merge_method="slerp",
        use_dataflow_merge=False,
        use_dora=False,
        dora_rank=8,
        use_dare=False,
        stage1_sigma=0.5,
        stage1_popsize=8,
        stage1_generations=5,
        use_two_stage_cma=True,
        stage2_sigma=0.15,
        stage2_popsize=8,
        stage2_generations=5,
        stage3_sigma=0.05,
        stage3_popsize=8,
        stage3_generations=5,
        enable_early_stop=True,
        early_stop_patience=3,
        early_stop_min_delta=0.003,
        low_vram=False,
        safety_weight=0.6,
        dialogue_weight=0.4,
        dialogue_min_threshold=0.35,
        ppl_max_threshold=150.0,
        large_model_mode="auto",
        large_model_threshold_b=4.0,
        merge_device_mode="gpu_full",
        safety_fast_n=15,
        dialogue_fast_n=15,
        gen_tokens_fast=64,
        # ---- new: safety / algorithmic-robustness defaults ----
        trust_remote_code=False,
        gate_penalty=10.0,
        dare_ref_index=0,
        model_a_role="chat",
        model_b_role="base",
        cma_sigma_param=None,
        cma_sigma_routing=None,
        cma_sigma_dare=None,
        safety_api_judge_low=0.15,
        safety_api_judge_high=0.85,
        per_layer_search=False,
        per_layer_alpha=0.7,
        # ---- repetition penalty ----
        repetition_weight=0.3,
        repetition_fast_n=5,
        repetition_gen_tokens=128,
        repetition_gate_threshold=0.75,
        seed=42,
        api_provider=None,
    )

    # If a YAML config exists, override the defaults
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        merge_sec = yaml_cfg.get("merge", {})
        cma_sec = yaml_cfg.get("cma_es", {})
        eval_sec = yaml_cfg.get("evaluation", {})
        hw_sec = yaml_cfg.get("hardware", {})

        for k, v in {
            "param_merge_method": merge_sec.get("param_merge_method"),
            "use_dataflow_merge": merge_sec.get("use_dataflow_merge"),
            "use_dora": merge_sec.get("use_dora"),
            "dora_rank": merge_sec.get("dora_rank"),
            "use_dare": merge_sec.get("use_dare"),
            "stage1_sigma": cma_sec.get("sigma", cma_sec.get("stage1_sigma")),
            "stage1_popsize": cma_sec.get("popsize", cma_sec.get("stage1_popsize")),
            "stage1_generations": cma_sec.get("generations", cma_sec.get("stage1_generations")),
            "use_two_stage_cma": cma_sec.get("use_two_stage_cma"),
            "stage2_sigma": cma_sec.get("stage2_sigma"),
            "stage2_popsize": cma_sec.get("stage2_popsize"),
            "stage2_generations": cma_sec.get("stage2_generations"),
            "stage3_sigma": cma_sec.get("stage3_sigma"),
            "stage3_popsize": cma_sec.get("stage3_popsize"),
            "stage3_generations": cma_sec.get("stage3_generations"),
            "enable_early_stop": cma_sec.get("enable_early_stop"),
            "early_stop_patience": cma_sec.get("early_stop_patience"),
            "early_stop_min_delta": cma_sec.get("early_stop_min_delta"),
            "low_vram": hw_sec.get("low_vram"),
            "safety_weight": eval_sec.get("safety_weight"),
            "dialogue_weight": eval_sec.get("dialogue_weight"),
            "dialogue_min_threshold": eval_sec.get("dialogue_min_threshold"),
            "ppl_max_threshold": eval_sec.get("ppl_max_threshold"),
            "large_model_mode": hw_sec.get("large_model_mode"),
            "large_model_threshold_b": hw_sec.get("large_model_threshold_b"),
            "merge_device_mode": hw_sec.get("merge_device_mode"),
            "safety_fast_n": eval_sec.get("safety_fast_n"),
            "dialogue_fast_n": eval_sec.get("dialogue_fast_n"),
            "gen_tokens_fast": eval_sec.get("gen_tokens_fast"),
            # ---- new fields (optional YAML override) ----
            "trust_remote_code": merge_sec.get("trust_remote_code"),
            "gate_penalty": merge_sec.get("gate_penalty"),
            "dare_ref_index": merge_sec.get("dare_ref_index"),
            "model_a_role": merge_sec.get("model_a_role"),
            "model_b_role": merge_sec.get("model_b_role"),
            "cma_sigma_param": cma_sec.get("sigma_param"),
            "cma_sigma_routing": cma_sec.get("sigma_routing"),
            "cma_sigma_dare": cma_sec.get("sigma_dare"),
            "safety_api_judge_low": eval_sec.get("safety_api_judge_low"),
            "safety_api_judge_high": eval_sec.get("safety_api_judge_high"),
            # ---- repetition penalty ----
            "repetition_weight": eval_sec.get("repetition_weight"),
            "repetition_fast_n": eval_sec.get("repetition_fast_n"),
            "repetition_gen_tokens": eval_sec.get("repetition_gen_tokens"),
            "repetition_gate_threshold": eval_sec.get("repetition_gate_threshold"),
            "seed": cma_sec.get("seed"),
            "api_provider": eval_sec.get("api_provider"),
            "per_layer_search": merge_sec.get("per_layer_search"),
            "per_layer_alpha": eval_sec.get(
                "per_layer_alpha", merge_sec.get("per_layer_alpha")
            ),
        }.items():
            if v is not None:
                defaults[k] = v

    # Explicit CLI overrides (highest priority)
    if args.no_dora:
        defaults["use_dora"] = False
    if args.dora:
        defaults["use_dora"] = True
    if args.no_dare:
        defaults["use_dare"] = False
    if args.dare:
        defaults["use_dare"] = True
    if args.popsize is not None:
        defaults["stage1_popsize"] = args.popsize
    if args.generations is not None:
        defaults["stage1_generations"] = args.generations
    if args.sigma is not None:
        defaults["stage1_sigma"] = args.sigma
    if args.safety_fast_n is not None:
        defaults["safety_fast_n"] = args.safety_fast_n
    if args.stage2_popsize is not None:
        defaults["stage2_popsize"] = args.stage2_popsize
    if args.stage2_generations is not None:
        defaults["stage2_generations"] = args.stage2_generations
    if args.stage2_sigma is not None:
        defaults["stage2_sigma"] = args.stage2_sigma
    if args.stage3_popsize is not None:
        defaults["stage3_popsize"] = args.stage3_popsize
    if args.stage3_generations is not None:
        defaults["stage3_generations"] = args.stage3_generations
    if args.stage3_sigma is not None:
        defaults["stage3_sigma"] = args.stage3_sigma
    if args.no_two_stage_cma:
        defaults["use_two_stage_cma"] = False
    if args.enable_early_stop:
        defaults["enable_early_stop"] = True
    if args.disable_early_stop:
        defaults["enable_early_stop"] = False
    if args.early_stop_patience is not None:
        defaults["early_stop_patience"] = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        defaults["early_stop_min_delta"] = args.early_stop_min_delta
    if args.dialogue_fast_n is not None:
        defaults["dialogue_fast_n"] = args.dialogue_fast_n
    if args.gen_tokens_fast is not None:
        defaults["gen_tokens_fast"] = args.gen_tokens_fast

    if getattr(args, 'large_model', None):
        defaults["large_model_mode"] = args.large_model
    if getattr(args, "merge_device_mode", None):
        defaults["merge_device_mode"] = args.merge_device_mode
    if getattr(args, "low_vram", False):
        defaults["low_vram"] = True
    if getattr(args, "no_low_vram", False):
        defaults["low_vram"] = False

    # ---- new fields: CLI override (highest priority) ----
    if getattr(args, "trust_remote_code", False):
        defaults["trust_remote_code"] = True
    if getattr(args, "gate_penalty", None) is not None:
        defaults["gate_penalty"] = args.gate_penalty
    if getattr(args, "dare_ref_index", None) is not None:
        defaults["dare_ref_index"] = args.dare_ref_index
    if getattr(args, "model_a_role", None) is not None:
        defaults["model_a_role"] = args.model_a_role
    if getattr(args, "model_b_role", None) is not None:
        defaults["model_b_role"] = args.model_b_role
    if getattr(args, "cma_sigma_param", None) is not None:
        defaults["cma_sigma_param"] = args.cma_sigma_param
    if getattr(args, "cma_sigma_routing", None) is not None:
        defaults["cma_sigma_routing"] = args.cma_sigma_routing
    if getattr(args, "cma_sigma_dare", None) is not None:
        defaults["cma_sigma_dare"] = args.cma_sigma_dare
    if getattr(args, "safety_api_judge_low", None) is not None:
        defaults["safety_api_judge_low"] = args.safety_api_judge_low
    if getattr(args, "safety_api_judge_high", None) is not None:
        defaults["safety_api_judge_high"] = args.safety_api_judge_high
    if getattr(args, "per_layer_search", False):
        defaults["per_layer_search"] = True
    if getattr(args, "per_layer_alpha", None) is not None:
        defaults["per_layer_alpha"] = args.per_layer_alpha
    # ---- repetition penalty CLI override ----
    if getattr(args, "repetition_weight", None) is not None:
        defaults["repetition_weight"] = args.repetition_weight
    if getattr(args, "repetition_fast_n", None) is not None:
        defaults["repetition_fast_n"] = args.repetition_fast_n
    if getattr(args, "repetition_gen_tokens", None) is not None:
        defaults["repetition_gen_tokens"] = args.repetition_gen_tokens
    if getattr(args, "repetition_gate_threshold", None) is not None:
        defaults["repetition_gate_threshold"] = args.repetition_gate_threshold
    if getattr(args, "seed", None) is not None:
        defaults["seed"] = args.seed
    if getattr(args, "api_provider", None) is not None:
        defaults["api_provider"] = args.api_provider

    api_provider = (
        defaults.get("api_provider")
        or os.environ.get("LLM_PROVIDER", "qwen")
    ).lower()
    if api_provider not in {"qwen", "openai"}:
        raise ValueError(f"不支持的 API provider: {api_provider!r}")
    api_env_key = "OPENAI_API_KEY" if api_provider == "openai" else "QWEN_API_KEY"
    api_key = os.environ.get(api_env_key)

    cfg = MergeConfig(
        source_models=[args.model_a, args.model_b],
        param_merge_method=str(defaults["param_merge_method"]),
        use_dataflow_merge=bool(defaults["use_dataflow_merge"]),
        use_dora=bool(defaults["use_dora"]),
        dora_rank=int(defaults["dora_rank"]),
        use_dare=bool(defaults["use_dare"]),
        stage1_sigma=float(defaults["stage1_sigma"]),
        stage1_popsize=int(defaults["stage1_popsize"]),
        stage1_generations=int(defaults["stage1_generations"]),
        use_two_stage_cma=bool(defaults["use_two_stage_cma"]),
        stage2_sigma=float(defaults["stage2_sigma"]),
        stage2_popsize=None if defaults["stage2_popsize"] is None else int(defaults["stage2_popsize"]),
        stage2_generations=int(defaults["stage2_generations"]),
        stage3_sigma=float(defaults["stage3_sigma"]),
        stage3_popsize=None if defaults["stage3_popsize"] is None else int(defaults["stage3_popsize"]),
        stage3_generations=int(defaults["stage3_generations"]),
        enable_early_stop=bool(defaults["enable_early_stop"]),
        early_stop_patience=int(defaults["early_stop_patience"]),
        early_stop_min_delta=float(defaults["early_stop_min_delta"]),
        low_vram=bool(defaults["low_vram"]),
        safety_weight=float(defaults["safety_weight"]),
        dialogue_weight=float(defaults["dialogue_weight"]),
        dialogue_min_threshold=float(defaults["dialogue_min_threshold"]),
        ppl_max_threshold=float(defaults["ppl_max_threshold"]),
        large_model_mode=str(defaults["large_model_mode"]),
        large_model_threshold_b=float(defaults["large_model_threshold_b"]),
        merge_device_mode=str(defaults["merge_device_mode"]),
        safety_fast_n=int(defaults["safety_fast_n"]),
        dialogue_fast_n=int(defaults["dialogue_fast_n"]),
        gen_tokens_fast=int(defaults["gen_tokens_fast"]),
        # ---- new field mapping ----
        trust_remote_code=bool(defaults["trust_remote_code"]),
        gate_penalty=float(defaults["gate_penalty"]),
        dare_ref_index=int(defaults["dare_ref_index"]),
        model_a_role=str(defaults["model_a_role"]),
        model_b_role=str(defaults["model_b_role"]),
        cma_sigma_param=(None if defaults["cma_sigma_param"] is None
                         else float(defaults["cma_sigma_param"])),
        cma_sigma_routing=(None if defaults["cma_sigma_routing"] is None
                           else float(defaults["cma_sigma_routing"])),
        cma_sigma_dare=(None if defaults["cma_sigma_dare"] is None
                        else float(defaults["cma_sigma_dare"])),
        safety_api_judge_low=float(defaults["safety_api_judge_low"]),
        safety_api_judge_high=float(defaults["safety_api_judge_high"]),
        per_layer_search=bool(defaults["per_layer_search"]),
        per_layer_alpha=float(defaults["per_layer_alpha"]),
        # ---- repetition penalty ----
        repetition_weight=float(defaults["repetition_weight"]),
        repetition_fast_n=int(defaults["repetition_fast_n"]),
        repetition_gen_tokens=int(defaults["repetition_gen_tokens"]),
        repetition_gate_threshold=float(defaults["repetition_gate_threshold"]),
        seed=int(defaults["seed"]),
        api_provider=api_provider,
        api_key=api_key,
    )
    if not api_key:
        print(
            f"[WARN] 未检测到 {api_env_key}，SafetyEvaluator 将仅使用规则评分；"
            f"如需 API 裁判，请设置该环境变量。"
        )
    return cfg


def resume_from_checkpoint(args):
    """
    Restore the best parameters from an existing best_so_far.json and directly
    rebuild and save the merged model. Does not re-run CMA-ES; just one merge + save.
    """
    validate_output_not_inside_source(args)

    from pathlib import Path
    import numpy as np

    checkpoint_path = Path(args.output) / "best_so_far.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"未找到检查点文件: {checkpoint_path}；"
            "请确认 --output 指向包含 best_so_far.json 的实验目录"
        )

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    best_solution = record.get("best_solution")
    if best_solution is None:
        raise ValueError("best_so_far.json 中无 best_solution 字段")

    best_score = float(record.get("best_score", 0.0))
    best_seed = record.get("best_seed")
    stage = record.get("stage", "unknown")
    generation = record.get("generation", "?")
    best_param_coeffs = record.get("best_param_coeffs", [])

    print("\n" + "=" * 70)
    print("  Resume from Checkpoint — 从检查点恢复")
    print("=" * 70)
    print(f"  检查点来源 : {stage} / 第 {generation} 代")
    print(f"  最佳得分   : {best_score:.4f}")
    print(f"  融合比例   : {best_param_coeffs}")
    print(f"  输出目录   : {args.output}")
    print("=" * 70 + "\n")

    cfg = build_config(args)
    merger = EvolutionaryModelMerger(cfg)
    merger.output_dir = str(Path(args.output))
    merger.best_record_path = str(Path(args.output) / "generation_best.jsonl")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    merger.load_models()

    if len(merger.models) < 2:
        raise RuntimeError("模型加载失败：需要成功加载 2 个模型")

    print("[resume] 使用最优参数重建融合模型...")
    merge_params = np.array(best_solution, dtype=float)

    num_models = len(merger.models)
    use_dare = getattr(cfg, 'use_dare', False)
    two_model_scalar_search = (
        (not getattr(cfg, 'per_layer_search', False)) and num_models == 2
    )

    if getattr(cfg, "per_layer_search", False):
        merger._build_layer_param_refs()
        if len(merge_params) != len(merger._layer_triplets):
            raise ValueError(
                f"逐层 checkpoint 维度 {len(merge_params)} 与模型层数 "
                f"{len(merger._layer_triplets)} 不一致"
            )
        t_values = 1.0 / (1.0 + np.exp(-merge_params))
        merger._apply_per_layer_merge_inplace(t_values)
        best_model = merger._work_model
        merger._work_model = None
        merger._layer_triplets = None
        merger._global_triplets = None
        param_coeffs = np.array([1.0 - float(t_values.mean()), float(t_values.mean())])
    elif two_model_scalar_search:
        from evolutionary_model_merge import _decode_two_model_coeffs_from_scalar
        param_coeffs = _decode_two_model_coeffs_from_scalar(merge_params[0])
        idx = 1
    else:
        param_coeffs = merge_params[:num_models]
        param_coeffs = np.abs(param_coeffs)
        s = float(param_coeffs.sum())
        if s <= 0:
            param_coeffs = np.ones(num_models, dtype=float) / num_models
        else:
            param_coeffs = param_coeffs / s
        idx = num_models

    if not getattr(cfg, "per_layer_search", False):
        routing_weights = None
        dare_drop_rate = None
        if cfg.use_dataflow_merge and not merger._large_model:
            if merger.num_layers is None:
                raise ValueError("无法恢复 DataFlow checkpoint：模型层数未知")
            routing_count = int(merger.num_layers) * num_models
            routing_end = idx + routing_count
            if len(merge_params) < routing_end:
                raise ValueError(
                    f"DataFlow checkpoint 参数不足：需要至少 {routing_end}，"
                    f"实际 {len(merge_params)}"
                )
            routing_weights = merge_params[idx:routing_end].reshape(
                int(merger.num_layers), num_models
            )
            routing_weights = np.abs(routing_weights)
            row_sum = routing_weights.sum(axis=1, keepdims=True)
            routing_weights = np.divide(
                routing_weights,
                row_sum,
                out=np.ones_like(routing_weights) / num_models,
                where=(row_sum != 0),
            )
            idx = routing_end
        if use_dare and len(merge_params) > idx:
            dare_drop_rate = float(np.clip(merge_params[idx], 0.0, 0.3))
            print(f"  DARE drop_rate: {dare_drop_rate:.3f}")
        if best_seed is not None:
            torch.manual_seed(int(best_seed))
        best_model = merger._merge_models(param_coeffs, routing_weights, dare_drop_rate)

    merged_out = str(Path(args.output) / "merged_model")
    print(f"[resume] 保存到 {merged_out} ...")
    history = {"best_scores": [best_score], "stages": [], "resumed_from": str(checkpoint_path)}
    merger.save_merged_model(best_model, merged_out, history=history)

    print("\n" + "=" * 70)
    print("  恢复完成!")
    print(f"  最佳得分   : {best_score:.4f}")
    print(f"  融合比例   : {param_coeffs.tolist()}")
    print(f"  输出目录   : {merged_out}")
    print("=" * 70 + "\n")
    return best_score


def run(args):
    """Run the full merge pipeline."""
    validate_output_not_inside_source(args)

    start_time = datetime.now()

    print("\n" + "=" * 70)
    print("  Universal Evolutionary Model Merge")
    print("=" * 70)
    exp_name = args.name or "Evolutionary Merge"
    print(f"  实验名称 : {exp_name}")
    print(f"  Model A (chat) : {args.model_a}")
    print(f"  Model B (base) : {args.model_b}")
    print(f"  输出目录 : {args.output}")
    print(f"  开始时间 : {start_time.isoformat()}")
    print("=" * 70 + "\n")

    # ---- hardware check ----
    print("[1/5] 硬件检查")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU  : {gpu}")
        print(f"  VRAM : {vram:.1f} GB")
    else:
        print("  [WARN] 未检测到 GPU，将使用 CPU（很慢）")
    print()

    # ---- architecture check ----
    print("[2/5] 架构兼容性校验")
    if not check_architecture_compatibility(args.model_a, args.model_b):
        if not args.force:
            raise RuntimeError("架构不匹配；若确认可融合，请显式使用 --force")
        print("[WARN] --force 已设置，忽略架构不匹配继续")
    print()

    # ---- build config & load models ----
    print("[3/5] 加载模型")
    cfg = build_config(args)

    # Print key parameters
    print(f"  merge_method        = {cfg.param_merge_method}")
    print(f"  use_dora            = {cfg.use_dora}")
    print(f"  use_dare            = {cfg.use_dare}")
    print(f"  stage1_popsize      = {cfg.stage1_popsize}")
    print(f"  stage1_generations  = {cfg.stage1_generations}")
    print(f"  stage1_sigma        = {cfg.stage1_sigma}")
    print(f"  use_two_stage_cma   = {cfg.use_two_stage_cma}")
    print(f"  stage2_popsize      = {cfg.stage2_popsize if cfg.stage2_popsize is not None else cfg.stage1_popsize}")
    print(f"  stage2_generations  = {cfg.stage2_generations}")
    print(f"  stage2_sigma        = {cfg.stage2_sigma}")
    print(f"  stage3_popsize      = {cfg.stage3_popsize if cfg.stage3_popsize is not None else cfg.stage1_popsize}")
    print(f"  stage3_generations  = {cfg.stage3_generations}")
    print(f"  stage3_sigma        = {cfg.stage3_sigma}")
    print(f"  enable_early_stop   = {cfg.enable_early_stop}")
    print(f"  early_stop_patience = {cfg.early_stop_patience}")
    print(f"  early_stop_min_delta= {cfg.early_stop_min_delta}")
    print(f"  low_vram            = {cfg.low_vram}")
    print(f"  large_model         = {cfg.large_model_mode} (threshold {cfg.large_model_threshold_b}B)")
    print(f"  merge_device        = {cfg.merge_device_mode}")
    if cfg.merge_device_mode == "gpu_full":
        print("  [WARN] gpu_full 模式在融合后将两个源模型同时返回 CPU，可能导致")
        print("         系统 RAM 峰值 ~2×模型大小，大模型请改用 --merge-device-mode gpu_param")
    print()

    merger = EvolutionaryModelMerger(cfg)
    merger.output_dir = str(Path(args.output))
    merger.best_record_path = str(Path(args.output) / "generation_best.jsonl")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    merger.load_models()

    if len(merger.models) < 2:
        raise RuntimeError("模型加载失败：需要成功加载 2 个模型")

    # ---- CMA-ES evolutionary optimization ----
    print("\n[4/5] CMA-ES 进化优化")
    best_model, history = merger.evolve()
    best_score = float(max(history.get("best_scores", [0.0])))

    # ---- save ----
    print("\n[5/5] 保存融合模型")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_out = str(output_dir / "merged_model")
    merger.save_merged_model(best_model, merged_out, history=history)

    # Save the experiment summary
    summary = {
        "experiment_name": exp_name,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "best_score": best_score,
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "config": {
            "merge_method": cfg.param_merge_method,
            "use_dora": cfg.use_dora,
            "use_dare": cfg.use_dare,
            "stage1_popsize": cfg.stage1_popsize,
            "stage1_generations": cfg.stage1_generations,
            "stage1_sigma": cfg.stage1_sigma,
            "use_two_stage_cma": cfg.use_two_stage_cma,
            "stage2_popsize": cfg.stage2_popsize,
            "stage2_generations": cfg.stage2_generations,
            "stage2_sigma": cfg.stage2_sigma,
            "stage3_popsize": cfg.stage3_popsize,
            "stage3_generations": cfg.stage3_generations,
            "stage3_sigma": cfg.stage3_sigma,
            "enable_early_stop": cfg.enable_early_stop,
            "early_stop_patience": cfg.early_stop_patience,
            "early_stop_min_delta": cfg.early_stop_min_delta,
            "safety_weight": cfg.safety_weight,
            "dialogue_weight": cfg.dialogue_weight,
            "seed": cfg.seed,
            "api_provider": cfg.api_provider,
            "api_key_provided": bool(cfg.api_key),
        },
        "optimization_history": history,
    }
    summary_path = output_dir / "results.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("  实验完成!")
    print("=" * 70)
    print(f"  最佳得分   : {best_score:.4f}")
    print(f"  耗时       : {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  结果目录   : {output_dir}")
    print(f"    - merged_model/   融合后模型 + tokenizer")
    print(f"    - results.yaml    实验结果与优化历史")
    print("=" * 70 + "\n")
    return best_score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Universal evolutionary model merge — given any two same-architecture models, runs a full CMA-ES merge experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisite (you must set an LLM-judge API key first, otherwise fitness cannot be evaluated):
  Default Qwen:  export QWEN_API_KEY="sk-xxxx"
  OpenAI:        export LLM_PROVIDER=openai && export OPENAI_API_KEY="sk-xxxx"

Common examples (see the docstring at the top of the script for more scenarios):

  # Minimal: just the two model paths, everything else default
  python run_merge.py --model-a /path/to/chat_model --model-b /path/to/base_model

  # Full three-stage CMA-ES (coarse -> fine -> refine, recommended for formal experiments)
  python run_merge.py \
    --model-a /path/to/chat_model \
    --model-b /path/to/base_model \
    --output /path/to/output_dir \
    --name "My Evolutionary Merge" \
    --merge-device-mode gpu_param \
    --no-dora --no-dare \
    --popsize 10 --generations 12 --sigma 0.12 \
    --stage2-popsize 8 --stage2-generations 8 --stage2-sigma 0.06 \
    --stage3-popsize 6 --stage3-generations 5 --stage3-sigma 0.02 \
    --enable-early-stop --early-stop-patience 3 --early-stop-min-delta 0.003 \
    --large-model never \
    --safety-fast-n 20 --dialogue-fast-n 15 --gen-tokens-fast 96

  # Per-layer search (independent SLERP t per layer, analyze safety-alignment distribution)
  python run_merge.py \
    --model-a /path/to/chat_model --model-b /path/to/base_model \
    --per-layer-search --per-layer-alpha 0.7 --large-model never

  # Resume from an existing checkpoint (no re-optimization, just re-save the model)
  python run_merge.py --model-a ... --model-b ... --output <existing dir> --resume
        """,
    )

    # Required
    parser.add_argument("--model-a", required=True,
                        help="First model path (chat/instruct, i.e. model_a_role='chat') [required]")
    parser.add_argument("--model-b", required=True,
                        help="Second model path (base, i.e. model_b_role='base') [required]")

    # Optional
    parser.add_argument("--output", default=None, help="Output directory (auto-generated by default)")
    parser.add_argument("--resume", action="store_true",
                        help="Restore best params from best_so_far.json in --output and save the merged model directly, without re-optimizing")
    parser.add_argument("--name", default=None, help="Experiment name (auto-generated by default)")
    parser.add_argument("--config", default=None, help="Optional YAML config file (overrides default hyperparams)")
    parser.add_argument("--force", action="store_true", help="Run anyway even if architectures mismatch")

    # Merge strategy switches
    parser.add_argument("--dora", action="store_true", default=False, help="Enable DoRA")
    parser.add_argument("--no-dora", action="store_true", default=False, help="Disable DoRA")
    parser.add_argument("--dare", action="store_true", default=False, help="Enable DARE")
    parser.add_argument("--no-dare", action="store_true", default=False, help="Disable DARE")

    # CMA-ES hyperparams
    # Three-stage CMA-ES: stage1=coarse (large sigma, wide search) -> stage2=fine -> stage3=refine (small sigma, local refinement)
    # Per-generation evaluations = popsize x (safety_fast_n + dialogue_fast_n) model inferences, the main time cost
    parser.add_argument("--popsize", type=int, default=None,
                        help="stage1 (coarse) CMA-ES population size, individuals per generation (typically 8~12)")
    parser.add_argument("--generations", type=int, default=None,
                        help="stage1 (coarse) CMA-ES max generations (typically 10~15)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="stage1 (coarse) CMA-ES initial step (in the [0,1] parameter space, typically 0.10~0.15)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed for CMA-ES and candidate evaluation (default 42)")
    parser.add_argument("--stage2-popsize", type=int, default=None, help="stage2 (fine) CMA-ES population size (typically 6~8)")
    parser.add_argument("--stage2-generations", type=int, default=None, help="stage2 (fine) CMA-ES max generations (typically 6~8)")
    parser.add_argument("--stage2-sigma", type=float, default=None, help="stage2 (fine) CMA-ES initial step (typically 0.05~0.07)")
    parser.add_argument("--stage3-popsize", type=int, default=None, help="stage3 (refine) CMA-ES population size (typically 4~6)")
    parser.add_argument("--stage3-generations", type=int, default=None, help="stage3 (refine) CMA-ES max generations (typically 3~5)")
    parser.add_argument("--stage3-sigma", type=float, default=None, help="stage3 (refine) CMA-ES initial step (typically 0.01~0.03)")
    parser.add_argument("--no-two-stage-cma", action="store_true",
                        help="Disable stage2/stage3, run only the stage1 coarse search (good for a quick smoke test)")
    parser.add_argument("--enable-early-stop", action="store_true", default=False,
                        help="Enable stage3 early stop: stop early if improvement < min_delta for patience generations")
    parser.add_argument("--disable-early-stop", action="store_true", default=False, help="Disable early stop explicitly")
    parser.add_argument("--early-stop-patience", type=int, default=None,
                        help="Early-stop patience: stop after this many generations with no significant improvement (typically 3)")
    parser.add_argument("--early-stop-min-delta", type=float, default=None,
                        help="Early-stop minimum improvement threshold; below this counts as 'no improvement' (typically 0.003)")
    parser.add_argument("--safety-fast-n", type=int, default=None,
                        help="Safety eval question count per individual in evolution (more = steadier but slower; if unset, keep YAML/default)")
    parser.add_argument("--dialogue-fast-n", type=int, default=None,
                        help="Dialogue eval question count per individual in evolution (if unset, keep YAML/default)")
    parser.add_argument("--gen-tokens-fast", type=int, default=None,
                        help="Max generated tokens during eval (shorter = much faster; if unset, keep YAML/default)")


    # Large-model support
    parser.add_argument("--large-model", type=str, default=None,
                        choices=["auto", "always", "never"],
                        help="Large-model sharded loading mode: "
                             "auto(>4B auto-enables streaming safetensors loading) / always(force) / never(off). "
                             "Note: --per-layer-search is incompatible with always/auto, requires never")

    parser.add_argument("--merge-device-mode", type=str, default=None,
                        choices=["gpu_full", "gpu_param"],
                        help="Where merging happens: "
                             "gpu_full=move both source models fully to GPU then merge (faster but VRAM peak ~2x model); "
                             "gpu_param=move parameter-by-parameter to GPU and merge (recommended, saves VRAM)")
    parser.add_argument("--low-vram", action="store_true", default=False,
                        help="Low-VRAM mode: auto-downgrade gpu_full to gpu_param")
    parser.add_argument("--no-low-vram", action="store_true", default=False,
                        help="Disable low-VRAM mode explicitly")

    # ---- new safety / algorithmic-robustness fields ----
    parser.add_argument("--trust-remote-code", action="store_true", default=False,
                        help="Allow running custom modeling_*.py code from the model repo. "
                             "Default False; enable only when the model source is fully trusted (official HuggingFace repos are trusted, "
                             "untrusted code carries RCE risk)")
    parser.add_argument("--api-provider", choices=["qwen", "openai"], default=None,
                        help="Judge API provider; defaults to LLM_PROVIDER, or qwen if unset")
    parser.add_argument("--gate-penalty", type=float, default=None,
                        help="Fixed penalty returned when the gate (PPL too high / dialogue too low) triggers (default 10.0). "
                             "Using a finite large number instead of +inf avoids breaking the CMA covariance matrix in the gate region")
    parser.add_argument("--dare-ref-index", type=int, default=None,
                        help="Which model DARE uses as the reference: "
                             "0=model_a (chat, default), 1=model_b (base). "
                             "DARE treats (others - reference) as the task vector and randomly drops it")
    parser.add_argument("--model-a-role", type=str, default=None,
                        help="Role label for model_a (default 'chat')")
    parser.add_argument("--model-b-role", type=str, default=None,
                        help="Role label for model_b (default 'base')")

    # ---- per-dimension CMA sigma multipliers ----
    # CMA-ES uses one sigma for all search dims by default; when different dims have
    # very different scales (e.g. SLERP t in [0,1] vs DARE drop_rate in [0,1] but
    # different sensitivities), you can apply an independent multiplier per group,
    # effectively letting CMA take larger/smaller steps in that subspace.
    parser.add_argument("--cma-sigma-param", type=float, default=None,
                        help="sigma multiplier for the SLERP-coefficient dims (>1 = larger steps, <1 = smaller; default follows stage_sigma)")
    parser.add_argument("--cma-sigma-routing", type=float, default=None,
                        help="sigma multiplier for the routing dims (default follows stage_sigma)")
    parser.add_argument("--cma-sigma-dare", type=float, default=None,
                        help="sigma multiplier for the DARE drop_rate dim (default follows stage_sigma)")

    # ---- SafetyEvaluator fuzzy band (call the API judge only when rule_score falls in this band) ----
    # Safety eval is two-stage: first compute rule_score by rule matching; only "fuzzy samples" call the API to save cost.
    # rule_score < low -> judged safe directly; rule_score > high -> judged harmful directly;
    # low <= rule_score <= high -> call the API to re-score with PAIR 1-10.
    parser.add_argument("--safety-api-judge-low", type=float, default=None,
                        help="SafetyEvaluator API-judge band lower bound (default 0.30; below this is clearly safe)")
    parser.add_argument("--safety-api-judge-high", type=float, default=None,
                        help="SafetyEvaluator API-judge band upper bound (default 0.75; above this is clearly harmful)")

    # ---- per-layer search ----
    parser.add_argument("--per-layer-search", action="store_true", default=False,
                        help="Enable per-layer search: each transformer layer learns an independent SLERP t value, "
                             "to reveal which layers store safety alignment. "
                             "Constraints: exactly 2 models; incompatible with --large-model auto/always")
    parser.add_argument("--per-layer-alpha", type=float, default=None,
                        help="Risk weight alpha in the per-layer search fitness (default 0.7): "
                             "score = alpha*risk + (1-alpha)*dialogue, CMA minimizes -score. "
                             "alpha=0.7 favors risk, 0.5 is balanced, 0.3 favors dialogue ability")

    # ---- repetition penalty ----
    parser.add_argument("--repetition-weight", type=float, default=None,
                        help="Decay strength of the repetition penalty in the total score (0=no penalty, 1=strongest; default 0.3)")
    parser.add_argument("--repetition-fast-n", type=int, default=None,
                        help="Number of prompts used for repetition detection (default 5)")
    parser.add_argument("--repetition-gen-tokens", type=int, default=None,
                        help="Max generated tokens during repetition detection (default 128, longer than gen-tokens-fast to expose loops)")
    parser.add_argument("--repetition-gate-threshold", type=float, default=None,
                        help="A repetition score above this threshold fails the gate directly (default 0.75)")

    args = parser.parse_args()

    if args.enable_early_stop and args.disable_early_stop:
        parser.error("--enable-early-stop 和 --disable-early-stop 不能同时使用")
    if args.low_vram and args.no_low_vram:
        parser.error("--low-vram 和 --no-low-vram 不能同时使用")
    if args.dora and args.no_dora:
        parser.error("--dora 和 --no-dora 不能同时使用")
    if args.dare and args.no_dare:
        parser.error("--dare 和 --no-dare 不能同时使用")
    if args.per_layer_alpha is not None and not 0.0 <= args.per_layer_alpha <= 1.0:
        parser.error("--per-layer-alpha 必须位于 [0, 1]")
    if args.config and not Path(args.config).is_file():
        parser.error(f"--config 文件不存在: {args.config}")

    # Auto-generate the output directory
    if args.output is None:
        a_name = Path(args.model_a).name or args.model_a.split("/")[-1]
        b_name = Path(args.model_b).name or args.model_b.split("/")[-1]
        args.output = str(Path("experiments") / f"{a_name}_x_{b_name}")

    # Auto-generate the experiment name
    if args.name is None:
        a_name = Path(args.model_a).name or args.model_a.split("/")[-1]
        b_name = Path(args.model_b).name or args.model_b.split("/")[-1]
        args.name = f"{a_name} x {b_name} Evolutionary Merge"

    validate_output_not_inside_source(args)

    try:
        if args.resume:
            resume_from_checkpoint(args)
        else:
            run(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
