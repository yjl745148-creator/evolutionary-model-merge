#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-layer CMA-ES merge + layer-coefficient heatmap (hardened version)
=====================================================================
Improvements:
1) Works with different model structures for layer-parameter detection
   (no longer hard-codes the "model.layers." prefix).
2) Supports FAST/FULL dual-fidelity evaluation, greatly reducing search cost.
3) Supports a large-model cleanup policy (never/auto/always) to reduce OOM risk.
4) Evaluation failures auto-degrade and are flagged; training does not stop.
5) Keeps history/result/heatmap outputs for paper analysis.

Usage example:
python layer_heatmap.py \
  --model-a /path/to/chat_model \
  --model-b /path/to/base_model \
  --output /path/to/output_dir \
  --generations 20 \
  --popsize 8 \
  --alpha 0.7 \
  --safety-fast-n 10 \
  --safety-full-n 23 \
  --dialogue-fast-n 10 \
  --dialogue-full-n 15 \
  --gen-tokens-fast 64 \
  --gen-tokens-full 128 \
  --full-every 8 \
  --large-model never
"""

import argparse
import gc
import json
import math
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cma
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evolutionary_model_merge import (
    DialogueEvaluator,
    SafetyEvaluator,
    _CHUNK_THRESHOLD,
    _chunked_dot_product,
    _chunked_norm,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


# Fitness penalty returned on evaluation failure: aligned with the default
# evolutionary_model_merge.gate_penalty. Extreme values like 1e6 would break
# CMA-ES covariance adaptation; 10.0 is far above the normal fitness range
# [-1, 0] while not polluting the search direction.
FAILURE_PENALTY = 10.0


def _atomic_write_json(path, data) -> None:
    """Write to a tmp file then os.replace, so a mid-run crash never leaves half-written JSON."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_is_large_model(args) -> bool:
    if args.large_model == "always":
        return True
    if args.large_model == "never":
        return False
    # auto: simple heuristic (tune as needed)
    return ("8b" in args.model_a.lower()) or ("8b" in args.model_b.lower())


def maybe_empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup_model(model, strong: bool = False):
    """Release the GPU/CPU memory associated with a model.

    Note: the `del model` inside this function only unbinds *this function's
    local variable*; references held by the caller are not freed by it. To let
    the GC actually reclaim the model, the caller must set its own outer
    variable to None after the call (e.g. `model_a = None; cleanup_model(
    model_a, strong=...)` — order does not matter, the key is that the outer
    reference disappears). This function mainly does CPU relocation + gc.collect
    + cuda cache clearing.
    """
    if model is not None:
        try:
            if strong and torch.cuda.is_available():
                # strong cleanup: move back to CPU before deleting
                model.to("cpu")
        except Exception:
            pass
        del model
    gc.collect()
    maybe_empty_cuda_cache()



def resolve_api_provider_and_key(args):
    provider = (args.api_provider or os.environ.get("LLM_PROVIDER", "qwen")).lower()
    if provider not in {"qwen", "openai"}:
        provider = "qwen"
    env_key = "OPENAI_API_KEY" if provider == "openai" else "QWEN_API_KEY"
    return provider, (args.api_key or os.environ.get(env_key)), env_key

def get_transformer_layers(model):
    """
    Works with common CausalLM structures; returns the list of transformer blocks.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # LLaMA/Qwen
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT2
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers  # GPT-NeoX
    raise AttributeError("Unsupported model architecture: cannot find transformer layers")


def report_param_overlap(model_a, model_b):
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_cpu(path: str, trust_remote_code: bool = False):
    """Load a model onto CPU.

    - Validate that the path exists before from_pretrained, with a clear error.
    - trust_remote_code is passed explicitly by the caller (default False) to
      avoid RCE risk.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"模型路径不存在: {path} (请检查 --model-a / --model-b 参数)"
        )
    print(f"  加载: {path} (trust_remote_code={trust_remote_code})")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-layer SLERP
# ---------------------------------------------------------------------------

def slerp_tensors(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """SLERP interpolation; degrades to LERP when the angle is too small.

    - The shape-mismatch fallback also .to(orig_dtype) to avoid dtype drift.
    - norm/dot are chunked so we never hold two full fp32 copies of a/b at once.
    - sin uses math.sin (Python float), decoupled from torch tensors.
    """
    orig_dtype = a.dtype
    if a.shape != b.shape:
        return ((1 - t) * a + t * b).to(orig_dtype)
    a_flat = a.flatten()
    b_flat = b.flatten()
    na = _chunked_norm(a_flat)
    nb = _chunked_norm(b_flat)
    if float(na) < 1e-8 or float(nb) < 1e-8:
        return ((1 - t) * a + t * b).to(orig_dtype)
    dot_unit = _chunked_dot_product(a_flat, b_flat) / (na * nb)
    dot_unit = torch.clamp(dot_unit, -1.0, 1.0)
    theta = float(torch.acos(dot_unit))
    if abs(theta) < 1e-6:
        return ((1 - t) * a + t * b).to(orig_dtype)
    st = math.sin(theta)
    coef_a = math.sin((1 - t) * theta) / st
    coef_b = math.sin(t * theta) / st
    # Chunked weighted sum: only one chunk's fp32 copy is held at a time,
    # so peak memory is about chunk_size x 4B.
    n = a_flat.numel()
    result = torch.empty(n, dtype=orig_dtype, device=a.device)
    cs = _CHUNK_THRESHOLD
    for i in range(0, n, cs):
        j = min(i + cs, n)
        chunk = a_flat[i:j].float().mul_(coef_a).add_(
            b_flat[i:j].float(), alpha=coef_b)
        result[i:j] = chunk.to(orig_dtype)
        del chunk
    return result.reshape(a.shape)


def build_layer_param_refs(model_a, model_b, work_model):
    """
    Pre-build parameter references to avoid repeating named_parameters per eval.
    Returns:
      layer_triplets: List[List[(p_work, p_a, p_b)]]
      global_triplets: List[(p_work, p_a, p_b)]

    - Uses raise ValueError instead of assert (assert is stripped under python -O).
    - If a work_model per-layer parameter has no counterpart in a/b, raise
      immediately to avoid silently dropping params and producing a wrong model.
    """
    layers_a = list(get_transformer_layers(model_a))
    layers_b = list(get_transformer_layers(model_b))
    layers_w = list(get_transformer_layers(work_model))
    if not (len(layers_a) == len(layers_b) == len(layers_w)):
        raise ValueError(
            f"A/B/work 层数不一致: A={len(layers_a)}, B={len(layers_b)}, "
            f"work={len(layers_w)}"
        )

    layer_triplets = []
    layer_param_fullnames = set()

    # Find each layer's prefix in the full model (robust approach: reverse-lookup by id).
    nd_w = dict(work_model.named_parameters())
    name_of_param_w = {id(p): n for n, p in nd_w.items()}

    for layer_idx, (la, lb, lw) in enumerate(zip(layers_a, layers_b, layers_w)):
        pd_a = dict(la.named_parameters())
        pd_b = dict(lb.named_parameters())
        cur = []
        missing = []
        for n, p_w in lw.named_parameters():
            if n in pd_a and n in pd_b:
                cur.append((p_w, pd_a[n], pd_b[n]))
                # Record the work_model full name, used later to exclude from global.
                full_name = name_of_param_w.get(id(p_w), None)
                if full_name is not None:
                    layer_param_fullnames.add(full_name)
            else:
                missing.append(n)
        if missing:
            # Silently dropping params would make the layer mix incomplete and
            # produce a wrong model, so we must raise.
            raise ValueError(
                f"build_layer_param_refs: 第 {layer_idx} 层有 {len(missing)} 个 "
                f"work_model 参数在 A/B 中缺失,无法构建对齐三元组: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        layer_triplets.append(cur)

    nd_a = dict(model_a.named_parameters())
    nd_b = dict(model_b.named_parameters())
    nd_w = dict(work_model.named_parameters())

    global_triplets = []
    for n, p_w in nd_w.items():
        if n in layer_param_fullnames:
            continue
        if n in nd_a and n in nd_b:
            global_triplets.append((p_w, nd_a[n], nd_b[n]))

    return layer_triplets, global_triplets


@torch.no_grad()
def apply_per_layer_merge_inplace(layer_triplets, global_triplets, t_values: np.ndarray):
    """
    Write the merge result into work_model parameters in place.
    """
    n_layers = len(t_values)
    global_t = float(np.mean(t_values))

    # Per layer
    for i in range(n_layers):
        t_i = float(t_values[i])
        for p_w, p_a, p_b in layer_triplets[i]:
            p_w.data.copy_(slerp_tensors(p_a.data, p_b.data, t_i))

    # Global parameters
    for p_w, p_a, p_b in global_triplets:
        p_w.data.copy_(slerp_tensors(p_a.data, p_b.data, global_t))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_with_mode(work_model_cpu, tokenizer, safety_eval, dialogue_eval, args, mode: str):
    """
    mode: 'fast' or 'full'
    Returns: dict {safety, dialogue, asr, peak_mem_gb, mode, error}
    """
    assert mode in ("fast", "full")
    merged_gpu = None
    out = {
        "safety": None,
        "dialogue": None,
        "asr": None,
        "peak_mem_gb": None,
        "mode": mode,
        "error": None,
    }

    try:
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            torch.cuda.reset_peak_memory_stats()
            # deepcopy avoids .to() polluting work_model in place, keeping it always on CPU
            merged_gpu = deepcopy(work_model_cpu).to(device="cuda", dtype=dtype)
        else:
            merged_gpu = work_model_cpu  # CPU path needs no migration

        # Pick fast/full parameters based on mode
        if mode == "fast":
            fast_n_s = args.safety_fast_n
            full_n_s = args.safety_full_n
            fast_n_d = args.dialogue_fast_n
            full_n_d = args.dialogue_full_n
            gen_tokens = args.gen_tokens_fast
        else:
            fast_n_s = args.safety_fast_n
            full_n_s = args.safety_full_n
            fast_n_d = args.dialogue_fast_n
            full_n_d = args.dialogue_full_n
            gen_tokens = args.gen_tokens_full

        safety = float(safety_eval.evaluate(
            merged_gpu, tokenizer,
            mode=mode,
            fast_n=fast_n_s,
            full_n=full_n_s,
            max_new_tokens=gen_tokens,
        ))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        dialogue = float(dialogue_eval.evaluate(
            merged_gpu, tokenizer,
            mode=mode,
            fast_n=fast_n_d,
            full_n=full_n_d,
            max_new_tokens=gen_tokens,
        ))
        asr = 1.0 - safety

        out["safety"] = safety
        out["dialogue"] = dialogue
        out["asr"] = asr

        if torch.cuda.is_available():
            out["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)

    except Exception as e:
        out["error"] = str(e)

    finally:
        if merged_gpu is not None and merged_gpu is not work_model_cpu:
            # Release GPU memory: for large models move back to CPU before deleting, otherwise delete directly
            if args._is_large_model:
                try:
                    merged_gpu.to("cpu")
                except Exception:
                    pass
            del merged_gpu
        gc.collect()
        maybe_empty_cuda_cache()

    return out


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run(args):
    set_seed(args.seed)
    args._is_large_model = infer_is_large_model(args)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[配置]")
    print(f"  model_a              = {args.model_a}")
    print(f"  model_b              = {args.model_b}")
    print(f"  output               = {args.output}")
    print(f"  generations          = {args.generations}")
    print(f"  popsize              = {args.popsize}")
    print(f"  alpha                = {args.alpha}")
    print(f"  large_model          = {args.large_model} (resolved={args._is_large_model})")
    print(f"  safety fast/full     = {args.safety_fast_n}/{args.safety_full_n}")
    print(f"  dialogue fast/full   = {args.dialogue_fast_n}/{args.dialogue_full_n}")
    print(f"  gen tokens fast/full = {args.gen_tokens_fast}/{args.gen_tokens_full}")
    print(f"  full_every           = {args.full_every}")
    api_provider, api_key, api_env_key = resolve_api_provider_and_key(args)
    print(f"  api_provider         = {api_provider} (env fallback: {api_env_key})")

    trust_remote = bool(getattr(args, "trust_remote_code", False))
    if trust_remote:
        print("[WARN] trust_remote_code=True — 仅在确认模型来源可信时启用,否则存在 RCE 风险")
    else:
        print("[INFO] trust_remote_code=False (默认);若模型需要自定义代码,加 --trust-remote-code")

    print("\n[1/5] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] 加载源模型到 CPU ...")
    model_a = load_model_cpu(args.model_a, trust_remote_code=trust_remote)
    model_b = load_model_cpu(args.model_b, trust_remote_code=trust_remote)
    report_param_overlap(model_a, model_b)

    layers_a = list(get_transformer_layers(model_a))
    n_layers = len(layers_a)
    print(f"  检测到 {n_layers} 个 transformer 层 → 搜索空间: {n_layers}D")

    print("[3/5] 初始化 work_model（仅一次）...")
    work_model = deepcopy(model_a).cpu().eval()

    layer_triplets, global_triplets = build_layer_param_refs(model_a, model_b, work_model)
    print(f"  每层可融合参数组数量（示例第0层）: {len(layer_triplets[0]) if n_layers else 0}")
    print(f"  全局可融合参数组数量: {len(global_triplets)}")

    safety_eval = SafetyEvaluator(api_key=api_key, api_provider=api_provider)
    dialogue_eval = DialogueEvaluator()

    eval_count = [0]
    history = []
    alpha = float(args.alpha)

    best_seen = {
        "fitness": float("inf"),
        "t_values": None,
        "metrics_fast": None,
        "metrics_full": None,
    }

    def fitness(x_raw):
        t_values = 1.0 / (1.0 + np.exp(-np.asarray(x_raw)))

        eval_count[0] += 1
        n = eval_count[0]
        print(
            f"\n  [eval #{n}] t_mean={t_values.mean():.3f}  "
            f"t_min={t_values.min():.3f}  t_max={t_values.max():.3f}"
        )

        # The merge step itself can also raise (slerp numeric/dtype errors, etc.);
        # wrap in try/except so a failure goes to the same penalty branch as an
        # evaluation failure, avoiding dragging down the whole CMA-ES.
        try:
            apply_per_layer_merge_inplace(layer_triplets, global_triplets, t_values)
        except Exception as e:
            print(f"  [ERR][MERGE] {e}")
            score = FAILURE_PENALTY
            item = {
                "eval": n,
                "t_values": t_values.tolist(),
                "fitness": score,
                "mode": "merge_failed",
                "error": str(e),
            }
            history.append(item)
            return score

        # Default fast
        fast_m = evaluate_with_mode(work_model, tokenizer, safety_eval, dialogue_eval, args, "fast")
        if fast_m["error"] is not None:
            print(f"  [ERR][FAST] {fast_m['error']}")
            score = FAILURE_PENALTY  # large penalty on failure
            item = {
                "eval": n,
                "t_values": t_values.tolist(),
                "fitness": score,
                "mode": "fast",
                "error": fast_m["error"],
            }
            history.append(item)
            return score

        # Maximize (alpha*ASR + (1-alpha)*dialogue) -> minimize its negative
        score_fast = -(alpha * fast_m["asr"] + (1.0 - alpha) * fast_m["dialogue"])

        # Periodic full re-evaluation
        full_m = None
        use_full = (args.full_every > 0 and (n % args.full_every == 0))
        score = score_fast
        if use_full:
            full_m = evaluate_with_mode(work_model, tokenizer, safety_eval, dialogue_eval, args, "full")
            if full_m["error"] is None:
                score = -(alpha * full_m["asr"] + (1.0 - alpha) * full_m["dialogue"])
            else:
                print(f"  [WARN][FULL] {full_m['error']}，回退 FAST 分数")

        mem_info = ""
        if full_m and full_m["peak_mem_gb"] is not None:
            mem_info = f"  peak_mem(full)={full_m['peak_mem_gb']}GB"
        elif fast_m["peak_mem_gb"] is not None:
            mem_info = f"  peak_mem(fast)={fast_m['peak_mem_gb']}GB"

        print(
            f"  FAST: safety={fast_m['safety']:.4f} dialogue={fast_m['dialogue']:.4f} "
            f"ASR={fast_m['asr']:.4f} fitness={score_fast:.4f}"
        )
        if full_m and full_m["error"] is None:
            print(
                f"  FULL: safety={full_m['safety']:.4f} dialogue={full_m['dialogue']:.4f} "
                f"ASR={full_m['asr']:.4f} fitness={score:.4f}{mem_info}"
            )
        else:
            print(f"  USED fitness={score:.4f}{mem_info}")

        item = {
            "eval": n,
            "t_values": t_values.tolist(),
            "fitness": float(score),
            "fitness_fast": float(score_fast),
            "fast": fast_m,
            "full": full_m,
        }
        history.append(item)

        if score < best_seen["fitness"]:
            best_seen["fitness"] = float(score)
            best_seen["t_values"] = t_values.tolist()
            best_seen["metrics_fast"] = fast_m
            best_seen["metrics_full"] = full_m

        if n % 5 == 0:
            _atomic_write_json(output_dir / "history.json", history)

        # Large models can be cleaned up more aggressively
        if args._is_large_model:
            gc.collect()
            maybe_empty_cuda_cache()

        return float(score)

    # ---- CMA-ES ----
    x0 = np.zeros(n_layers)  # sigmoid(0)=0.5
    sigma0 = 0.5

    opts = cma.CMAOptions()
    opts["maxiter"] = args.generations
    opts["popsize"] = args.popsize
    opts["verbose"] = 1
    opts["seed"] = args.seed

    print(
        f"\n[4/5] CMA-ES 开始 ({n_layers}D, {args.generations} 代, "
        f"种群 {args.popsize}, alpha={alpha}) ..."
    )
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    while not es.stop():
        solutions = es.ask()
        fitnesses = [fitness(x) for x in solutions]
        es.tell(solutions, fitnesses)
        es.disp()

    _atomic_write_json(output_dir / "history.json", history)

    best_t = 1.0 / (1.0 + np.exp(-es.result.xbest))

    print("\n最优逐层混合比例 (t=0 纯chat, t=1 纯base):")
    for i, t in enumerate(best_t):
        bar = "█" * int(t * 20) + "░" * (20 - int(t * 20))
        print(f"  Layer {i:2d}: [{bar}] {t:.3f}")

    apply_per_layer_merge_inplace(layer_triplets, global_triplets, best_t)
    final_full_runs = [
        evaluate_with_mode(
            work_model, tokenizer, safety_eval, dialogue_eval, args, "full"
        )
    ]
    final_full = dict(final_full_runs[0])
    final_full["n_runs"] = 1

    result = {
        "experiment_time": datetime.now().isoformat(),
        "model_a": args.model_a,
        "model_b": args.model_b,
        "n_layers": int(n_layers),
        "alpha": alpha,
        "generations": args.generations,
        "popsize": args.popsize,
        "seed": args.seed,
        "large_model": args.large_model,
        "resolved_large_model": bool(args._is_large_model),
        "best_t_values": best_t.tolist(),
        "best_fitness": float(es.result.fbest),
        "total_evals": eval_count[0],
        "best_seen_during_search": best_seen,
        "final_full_metrics": final_full,
        "fast_full_config": {
            "safety_fast_n": args.safety_fast_n,
            "safety_full_n": args.safety_full_n,
            "dialogue_fast_n": args.dialogue_fast_n,
            "dialogue_full_n": args.dialogue_full_n,
            "gen_tokens_fast": args.gen_tokens_fast,
            "gen_tokens_full": args.gen_tokens_full,
            "full_every": args.full_every,
        },
    }
    result_path = output_dir / "result.json"
    _atomic_write_json(result_path, result)
    print(f"\n结果已保存: {result_path}")

    print("[5/5] 生成热力图 ...")
    try:
        plot_heatmap(best_t, history, output_dir)
    except Exception as e:
        print(f"  [WARN] 绘图失败: {e}（结果已保存，可手动绘图）")

    # After calling cleanup_model, set the outer reference to None right away so
    # the GC can actually reclaim it (cleanup_model's internal del only unbinds
    # the local parameter; see the function docstring).
    cleanup_model(work_model, strong=args._is_large_model)
    work_model = None
    cleanup_model(model_a, strong=False)
    model_a = None
    cleanup_model(model_b, strong=False)
    model_b = None
    gc.collect()
    maybe_empty_cuda_cache()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_heatmap(t_values: np.ndarray, history: list, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(t_values)
    fig = plt.figure(figsize=(16, max(5, n * 0.4)))
    gs = GridSpec(1, 3, width_ratios=[2, 1, 2], figure=fig)

    ax1 = fig.add_subplot(gs[0])
    data = t_values.reshape(-1, 1)
    im = ax1.imshow(data, cmap="coolwarm", aspect="auto", vmin=0, vmax=1)
    ax1.set_xticks([0])
    ax1.set_xticklabels(["Base Weight t"])
    ax1.set_yticks(range(n))
    ax1.set_yticklabels([f"Layer {i}" for i in range(n)], fontsize=8)
    ax1.set_title(
        "Per-Layer Merge Coefficient\n(blue=chat dominant, red=base dominant)",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax1, label="t (base weight)")

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t_values, range(n), "ko-", markersize=4, linewidth=1.5)
    ax2.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="t=0.5")
    ax2.fill_betweenx(
        range(n), t_values, 0.5,
        where=(t_values > 0.5), color="red", alpha=0.15, label="base dominant"
    )
    ax2.fill_betweenx(
        range(n), t_values, 0.5,
        where=(t_values < 0.5), color="blue", alpha=0.15, label="chat dominant"
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, n - 0.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Base Weight t")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([f"{i}" for i in range(n)], fontsize=7)
    ax2.set_title("t per Layer", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, loc="lower right")

    ax3 = fig.add_subplot(gs[2])
    evals = [h["eval"] for h in history]
    fitnesses = [h["fitness"] for h in history]
    asrs = []
    dialogues = []
    for h in history:
        if h.get("full") and h["full"].get("error") is None:
            asrs.append(h["full"]["asr"])
            dialogues.append(h["full"]["dialogue"])
        elif h.get("fast") and h["fast"].get("error") is None:
            asrs.append(h["fast"]["asr"])
            dialogues.append(h["fast"]["dialogue"])
        else:
            asrs.append(np.nan)
            dialogues.append(np.nan)

    ax3.plot(evals, fitnesses, "g-", linewidth=1, alpha=0.6, label="fitness (min)")
    ax3b = ax3.twinx()
    ax3b.plot(evals, asrs, "r--", linewidth=1.5, label="ASR")
    ax3b.plot(evals, dialogues, "b--", linewidth=1.5, label="Dialogue")
    ax3b.set_ylim(0, 1)
    ax3b.set_ylabel("ASR / Dialogue", fontsize=9)
    ax3.set_xlabel("Evaluation #")
    ax3.set_ylabel("Fitness", fontsize=9, color="g")
    ax3.set_title("Optimization Curve", fontsize=10)
    ax3.grid(True, alpha=0.3)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.suptitle(
        "Layer-wise CMA-ES Optimal Merge Coefficients\n"
        "Mechanistic Interpretability: Where is Safety Alignment Stored?",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()

    plot_path = output_dir / "layer_heatmap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  热力图已保存: {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Per-layer CMA-ES merge + layer-coefficient heatmap (hardened)")
    p.add_argument("--model-a", required=True, help="Path to the chat/instruct model")
    p.add_argument("--model-b", required=True, help="Path to the base model")
    p.add_argument("--output", default="experiments/layer_heatmap", help="Output directory")

    p.add_argument("--generations", type=int, default=20, help="CMA-ES max generations (default 20)")
    p.add_argument("--popsize", type=int, default=8, help="Population size (default 8)")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="fitness: -(alpha*ASR + (1-alpha)*dialogue); maximizes ASR while keeping dialogue, default 0.7 favors attack",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")

    # FAST/FULL
    p.add_argument("--safety-fast-n", type=int, default=10, help="FAST safety sample count")
    p.add_argument("--safety-full-n", type=int, default=23, help="FULL safety sample count")
    p.add_argument("--dialogue-fast-n", type=int, default=10, help="FAST dialogue sample count")
    p.add_argument("--dialogue-full-n", type=int, default=15, help="FULL dialogue sample count")
    p.add_argument("--gen-tokens-fast", type=int, default=64, help="FAST generation length")
    p.add_argument("--gen-tokens-full", type=int, default=128, help="FULL generation length")
    p.add_argument("--full-every", type=int, default=8, help="Run a FULL eval every N evals (0=off)")
    p.add_argument("--api-provider", default=None, help="Safety evaluator API provider (default: LLM_PROVIDER or qwen)")
    p.add_argument("--api-key", default=None, help="Safety evaluator API key (default: QWEN_API_KEY or OPENAI_API_KEY by provider)")

    # Cleanup policy
    p.add_argument(
        "--large-model",
        choices=["never", "auto", "always"],
        default="auto",
        help="Large-model cleanup policy: never/auto/always",
    )

    # Security: trust_remote_code is off by default; enable explicitly to run custom code from the model repo
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Allow transformers to run custom code from the model repo (off by default; confirm the source is trusted before enabling)",
    )

    return p.parse_args()


def _validate_args(args) -> None:
    """CLI argument sanity check; raise on invalid values to avoid weird behavior mid-training."""
    if args.generations < 1:
        raise ValueError(f"--generations 必须 >= 1,当前 {args.generations}")
    if args.popsize < 2:
        raise ValueError(f"--popsize 必须 >= 2(CMA-ES 需要至少 2 个候选),当前 {args.popsize}")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(f"--alpha 必须 ∈ [0, 1],当前 {args.alpha}")
    if args.full_every < 0:
        raise ValueError(f"--full-every 必须 >= 0(0 表示关闭),当前 {args.full_every}")


if __name__ == "__main__":
    args = parse_args()
    _validate_args(args)
    run(args)
