#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dialogue-only retest for an IT/chat model using a DeepSeek V4 Pro judge.

This script mirrors the dialogue portion of compare_models.py:
  - MT-Bench sampling via load_mtbench_prompts
  - AlpacaEval sampling via load_alpacaeval_prompts
  - response generation via generate
  - dialogue scoring rubric from qwen_score_dialogue

It intentionally skips harmful/safety evaluation and records every sample with
the input prompt, model response, raw judge output, parsed score, timestamps,
configuration, and any per-sample error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_models import (  # noqa: E402
    PROMPT_MODE_DESCRIPTIONS,
    compute_stats,
    generate,
    load_alpacaeval_prompts,
    load_model,
    load_mtbench_prompts,
    set_global_seed,
    unload_model,
)


DEFAULT_JUDGE_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV_CANDIDATES = (
    "DEEPSEEK_API_KEY",
    "QWEN_API_KEY",
    "DASHSCOPE_API_KEY",
    "JUDGE_API_KEY",
)

SCORING_USER_TEMPLATE = (
    "You are an expert evaluator for language model responses. "
    "Rate responses strictly and objectively.\n\n"
    "Please rate the following response to a [{category}] question on a scale of 0-10.\n\n"
    "Scoring criteria:\n"
    "- Relevance & Accuracy (0-3): Does the response correctly address the question?\n"
    "- Fluency & Coherence (0-3): Is the language natural, logical, and well-structured?\n"
    "- Depth & Informativeness (0-2): Does it provide substantive, insightful content?\n"
    "- Format & Presentation (0-2): Is the format appropriate and well-organized?\n\n"
    "Question: {question}\n\n"
    "Response: {answer}\n\n"
    "Output ONLY a single integer from 0 to 10. No explanation."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_source_report(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    report_path = Path(path)
    return json.loads(report_path.read_text(encoding="utf-8"))


def resolve_model_path(args: argparse.Namespace, source_report: dict[str, Any]) -> str:
    if args.model_path:
        return args.model_path
    report_model = source_report.get("models", {}).get("chat")
    if report_model:
        return str(report_model)
    raise SystemExit("ERROR: provide --model-path or --source-report containing models.chat")


def resolve_prompt_mode(args: argparse.Namespace, source_report: dict[str, Any]) -> str:
    if args.prompt_mode:
        return args.prompt_mode
    mode = source_report.get("meta", {}).get("prompt_mode")
    if mode in PROMPT_MODE_DESCRIPTIONS:
        return mode
    return "all_raw_no_system"


def prompt_policy(prompt_mode: str) -> bool:
    if prompt_mode == "all_raw_no_system":
        return False
    if prompt_mode == "chat_template_no_system":
        return True
    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


def resolve_count(cli_value: Optional[int], source_report: dict[str, Any], key: str, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    value = source_report.get("meta", {}).get(key)
    if value is not None:
        return int(value)
    return default


def resolve_api_key(args: argparse.Namespace) -> tuple[Optional[str], Optional[str]]:
    candidates = (args.api_key_env,) if args.api_key_env else API_KEY_ENV_CANDIDATES
    for env_name in candidates:
        if not env_name:
            continue
        value = os.environ.get(env_name)
        if value:
            return value, env_name
    return None, None


def resolve_judge_config(args: argparse.Namespace) -> dict[str, Any]:
    api_key, api_key_env = resolve_api_key(args)
    explicit_base = args.base_url or os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("JUDGE_BASE_URL")
    explicit_model = args.judge_model or os.environ.get("DEEPSEEK_JUDGE_MODEL") or os.environ.get("JUDGE_MODEL")
    if api_key is None and not args.api_key_env:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            api_key = openai_key
            api_key_env = "OPENAI_API_KEY"
            explicit_base = explicit_base or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            explicit_model = explicit_model or "gpt-4o-mini"
    return {
        "model": explicit_model or DEFAULT_JUDGE_MODEL,
        "base_url": explicit_base or DEFAULT_BASE_URL,
        "api_key": api_key,
        "api_key_env": api_key_env,
    }


def make_client(judge_config: dict[str, Any]):
    if not judge_config.get("api_key"):
        names = ", ".join(API_KEY_ENV_CANDIDATES)
        raise SystemExit(f"ERROR: no judge API key found. Set one of: {names}")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("ERROR: install openai first: pip install openai") from exc
    return OpenAI(api_key=judge_config["api_key"], base_url=judge_config["base_url"])


def call_judge_raw(
    client: Any,
    judge_model: str,
    question: str,
    answer: str,
    category: str,
    max_retries: int,
    request_timeout: Optional[float],
) -> str:
    user_prompt = SCORING_USER_TEMPLATE.format(
        category=category,
        question=question,
        answer=answer,
    )
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            stream = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=8,
                stream=True,
                timeout=request_timeout,
            )
            parts: list[str] = []
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = getattr(chunk.choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                if isinstance(content, str) and content:
                    parts.append(content)
            return "".join(parts).strip()
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"judge call failed: {last_error}")


def parse_dialogue_score(raw: str) -> Optional[float]:
    match = re.search(r"\b(10|[0-9])\b", raw or "")
    if not match:
        return None
    score = float(match.group(1))
    return min(max(score, 0.0), 10.0)


def strip_system_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_system_fields(item)
            for key, item in value.items()
            if key
            not in {
                "system_prompt",
                "system_role_inserted",
                "prompt_mode_description",
            }
        }
    if isinstance(value, list):
        return [strip_system_fields(item) for item in value]
    return value


def build_samples(mtbench_n: int, alpaca_n: int, seed: int, dataset: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if dataset in {"both", "mtbench"}:
        for item in load_mtbench_prompts(mtbench_n, seed):
            samples.append(
                {
                    "dataset": "mtbench",
                    "id": item["id"],
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "max_new_tokens": 400,
                }
            )
    if dataset in {"both", "alpaca"}:
        for item in load_alpacaeval_prompts(alpaca_n, seed):
            samples.append(
                {
                    "dataset": "alpaca",
                    "id": item["id"],
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "max_new_tokens": 300,
                }
            )
    return samples


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [item for item in items if item.get("score") is not None]
    scores_all = [float(item["score"]) for item in scored]
    scores_mtb = [float(item["score"]) for item in scored if item.get("dataset") == "mtbench"]
    scores_alpaca = [float(item["score"]) for item in scored if item.get("dataset") == "alpaca"]

    category_scores: dict[str, list[float]] = {}
    alpaca_category_scores: dict[str, list[float]] = {}
    for item in scored:
        bucket = category_scores if item.get("dataset") == "mtbench" else alpaca_category_scores
        bucket.setdefault(str(item.get("category", "unknown")), []).append(float(item["score"]))

    return {
        "stats": compute_stats(scores_all),
        "avg_score": compute_stats(scores_all)["mean"],
        "mtbench_stats": compute_stats(scores_mtb),
        "mtbench_avg": compute_stats(scores_mtb)["mean"],
        "alpaca_stats": compute_stats(scores_alpaca),
        "alpaca_avg": compute_stats(scores_alpaca)["mean"],
        "category_stats": {key: compute_stats(value) for key, value in category_scores.items()},
        "category_avg": {key: compute_stats(value)["mean"] for key, value in category_scores.items()},
        "alpaca_category_stats": {
            key: compute_stats(value) for key, value in alpaca_category_scores.items()
        },
        "total_samples": len(items),
        "scored_samples": len(scored),
        "failed_samples": sum(1 for item in items if item.get("error")),
        "parse_failed_samples": sum(
            1 for item in items if not item.get("error") and item.get("score") is None
        ),
        "empty_response_samples": sum(1 for item in items if not item.get("model_response")),
    }


def make_report_meta(
    args: argparse.Namespace,
    source_report: dict[str, Any],
    model_path: str,
    prompt_mode: str,
    use_chat_template: bool,
    judge_config: dict[str, Any],
    mtbench_n: int,
    alpaca_n: int,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_type": "it_dialogue_deepseek_retest",
        "created_at": utc_now(),
        "source_report": args.source_report,
        "source_report_meta": strip_system_fields(source_report.get("meta", {})),
        "source_report_models": source_report.get("models", {}),
        "tested_model": {
            "role": "chat/it",
            "model_path": model_path,
            "model_id": args.model_id or model_path,
        },
        "judge": {
            "model": judge_config["model"],
            "base_url": judge_config["base_url"],
            "api_key_env": judge_config["api_key_env"],
        },
        "prompt_mode": prompt_mode,
        "prompt_mode_description": PROMPT_MODE_DESCRIPTIONS[prompt_mode],
        "prompt_policy": {
            "use_chat_template": use_chat_template,
        },
        "sampling": {
            "dataset": args.dataset,
            "seed": args.seed,
            "mtbench_n": mtbench_n,
            "alpaca_n": alpaca_n,
            "limit": args.limit,
            "actual_samples": len(samples),
        },
        "generation_config": {
            "logic_source": "1123/compare_models.py:generate",
            "do_sample": True,
            "temperature": 0.7,
            "max_length": 4096,
            "mtbench_max_new_tokens": 400,
            "alpaca_max_new_tokens": 300,
        },
        "judge_prompt": {
            "user_template": SCORING_USER_TEMPLATE,
            "max_tokens": 8,
            "temperature": 0.0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retest only IT/chat model dialogue with DeepSeek V4 Pro judge."
    )
    parser.add_argument(
        "--source-report",
        default=None,
        help="Existing comparison_report.json. Defaults model path/counts/prompt mode from it.",
    )
    parser.add_argument("--model-path", default=None, help="IT/chat model path override.")
    parser.add_argument("--model-id", default=None, help="Human-readable tested model identifier.")
    parser.add_argument(
        "--prompt-mode",
        choices=list(PROMPT_MODE_DESCRIPTIONS.keys()),
        default=None,
        help="Defaults to source report meta.prompt_mode, else all_raw_no_system.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mtbench-n", type=int, default=None)
    parser.add_argument("--alpaca-n", type=int, default=None)
    parser.add_argument("--dataset", choices=["both", "mtbench", "alpaca"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N sampled dialogue items.")
    parser.add_argument("--output", default=None, help="Final JSON report path.")
    parser.add_argument("--jsonl-output", default=None, help="Per-item JSONL path.")
    parser.add_argument("--judge-model", default=None, help="Default/env fallback: deepseek-v4-pro.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible judge base URL.")
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Read judge API key from this environment variable only.",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved samples/config without loading model.")
    parser.add_argument("--list-config", action="store_true", help="Print resolved config and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_report = load_source_report(args.source_report)
    model_path = resolve_model_path(args, source_report)
    prompt_mode = resolve_prompt_mode(args, source_report)
    use_chat_template = prompt_policy(prompt_mode)
    mtbench_n = resolve_count(args.mtbench_n, source_report, "mtbench_n", 80)
    alpaca_n = resolve_count(args.alpaca_n, source_report, "alpaca_n", 50)
    judge_config = resolve_judge_config(args)

    set_global_seed(args.seed)
    samples = build_samples(mtbench_n, alpaca_n, args.seed, args.dataset)
    if args.limit is not None:
        samples = samples[: max(0, int(args.limit))]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output or f"it_dialogue_deepseek_retest_{timestamp}.json")
    jsonl_path = Path(args.jsonl_output or output_path.with_suffix(".jsonl"))

    meta = make_report_meta(
        args=args,
        source_report=source_report,
        model_path=model_path,
        prompt_mode=prompt_mode,
        use_chat_template=use_chat_template,
        judge_config=judge_config,
        mtbench_n=mtbench_n,
        alpaca_n=alpaca_n,
        samples=samples,
    )

    if args.list_config or args.dry_run:
        preview = {
            "meta": meta,
            "output_path": str(output_path),
            "jsonl_output_path": str(jsonl_path),
            "sample_preview": samples[: min(5, len(samples))],
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        if args.list_config:
            return
        if args.dry_run:
            return

    client = make_client(judge_config)
    model, tokenizer = load_model(model_path)

    items: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            for index, sample in enumerate(samples, 1):
                started_at = utc_now()
                t0 = time.time()
                item: dict[str, Any] = {
                    "index": index,
                    "dataset": sample["dataset"],
                    "id": sample["id"],
                    "category": sample["category"],
                    "prompt": sample["prompt"],
                    "max_new_tokens": sample["max_new_tokens"],
                    "tested_model_id": meta["tested_model"]["model_id"],
                    "tested_model_path": model_path,
                    "prompt_mode": prompt_mode,
                    "use_chat_template": use_chat_template,
                    "judge_model": judge_config["model"],
                    "judge_base_url": judge_config["base_url"],
                    "started_at": started_at,
                }
                try:
                    response = generate(
                        model,
                        tokenizer,
                        sample["prompt"],
                        max_new_tokens=sample["max_new_tokens"],
                        use_chat_template=use_chat_template,
                    )
                    item["model_response"] = response
                    raw = call_judge_raw(
                        client=client,
                        judge_model=judge_config["model"],
                        question=sample["prompt"],
                        answer=response,
                        category=sample["category"],
                        max_retries=max(0, int(args.max_retries)),
                        request_timeout=args.request_timeout,
                    )
                    score = parse_dialogue_score(raw)
                    item.update(
                        {
                            "model_response": response,
                            "judge_raw": raw,
                            "score": score,
                            "parse_error": None if score is not None else "No integer 0-10 found in judge_raw.",
                            "error": None,
                        }
                    )
                except Exception as exc:
                    item.update(
                        {
                            "model_response": item.get("model_response", ""),
                            "judge_raw": item.get("judge_raw", ""),
                            "score": None,
                            "parse_error": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                item["ended_at"] = utc_now()
                item["duration_sec"] = round(time.time() - t0, 3)
                items.append(item)
                jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")
                jsonl.flush()

                score_text = "N/A" if item["score"] is None else f"{item['score']:.1f}"
                print(
                    f"[{index:03d}/{len(samples):03d}] {sample['dataset']} "
                    f"{sample['id']} score={score_text} error={item['error'] or '-'}"
                )
    finally:
        unload_model(model)

    report = {
        "meta": meta,
        "summary": summarize(items),
        "items": items,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Final report written: {output_path}")
    print(f"Per-item JSONL written: {jsonl_path}")


if __name__ == "__main__":
    main()
