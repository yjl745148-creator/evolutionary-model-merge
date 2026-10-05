# Evolutionary Model Merge (EMMA)

EMMA uses CMA-ES to search merging parameters for two compatible Hugging Face causal language models. The repository contains global and layer-wise merging implementations, safety and dialogue evaluation, fixed-ratio ablation experiments, and paper figure sources.

## Repository Layout

| Path | Contents |
|---|---|
| [code/](code/README.md) | Model merging, search, evaluation, and ablation scripts. |
| [Figure_Code/](Figure_Code/README.md) | Paper figure sources and search-result plotting. |
| [date/](date/README.md) | Stored experiment reports, search histories, and logs. |
| [requirements.txt](requirements.txt) | Dependencies for the experiment code. |
| [LICENSE](LICENSE) | Project license. |

All commands below are run from this `main/` directory. The experiment-record directory is named `date/`; evaluation prompt files use the separate path `code/data/eval/`.

## Setup

Use Python 3.10 or 3.11 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Provide compatible chat/instruct and base checkpoints. Model dimensions, layer structure, vocabulary, and embedding configuration must support the selected merging method. Full experiments generally require a CUDA GPU with sufficient memory.

For API-based judging, configure the selected provider. For example, in a Bash shell:

```bash
export LLM_PROVIDER=qwen
export QWEN_API_KEY="your-api-key"
```

Alternatively, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`. The merge-stage evaluator can fall back to rule-based scoring when no corresponding API key is available; this fallback is distinct from formal API-judge evaluation.

## Model Merging

Global search:

```bash
python code/run_merge.py --model-a /path/to/chat_model --model-b /path/to/base_model --output experiments/global_run --seed 42
```

Layer-wise search:

```bash
python code/run_merge.py --model-a /path/to/chat_model --model-b /path/to/base_model --output experiments/layer_run --per-layer-search --per-layer-alpha 0.7 --seed 42
```

For the layer-wise objective, `--per-layer-alpha` controls the risk weight in the base weighted term:

```text
alpha * (1 - safety_score) + (1 - alpha) * dialogue_score
```

The implementation then applies the PPL factor and, when enabled, the repetition factor. These search proxy scores are distinct from the formal evaluation metrics.

Use `--low-vram` for the low-memory configuration and `python code/run_merge.py --help` for additional options.

To rebuild a merged model from saved best parameters:

```bash
python code/run_merge.py --model-a /path/to/chat_model --model-b /path/to/base_model --output experiments/layer_run --resume
```

Here, `--resume` reads saved best parameters and exports the merged model without continuing CMA-ES optimization. Use the same source checkpoints and compatible settings as the original run.

## Evaluation Data

The current evaluation code resolves prompt files relative to `code/compare_models.py`:

```text
code/data/eval/
  mt_bench_questions.jsonl
  alpaca_eval.json
  advbench_behaviors.jsonl
  harmbench_behaviors.jsonl
  jailbreakbench_behaviors.jsonl
```

Prepare these files in the formats expected by the loaders in `compare_models.py`. Missing files trigger built-in prompt fallbacks, so provide the intended datasets and sample counts when reproducing a specific evaluation protocol.

Dataset sources:

- [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)
- [AdvBench](https://github.com/llm-attacks/llm-attacks)
- [HarmBench](https://github.com/centerforaisafety/HarmBench)
- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)

## Model Evaluation

Compare merged, chat, and base models:

```bash
python code/compare_models.py --merged experiments/global_run/merged_model --chat /path/to/chat_model --base /path/to/base_model --output experiments/global_run/comparison_report.json
```

The current comparison entry point uses each model's tokenizer chat template when available, with a raw-prompt fallback when no template exists, and injects no system prompt. Empty dialogue responses and failed dialogue-judge calls are excluded from the dialogue mean; an evaluation with no valid dialogue scores raises an error.

For dialogue-only reevaluation:

```bash
python code/retest_it_dialogue_deepseek.py --model-path /path/to/model --model-id model_to_retest --prompt-mode chat_template_no_system --judge-model YOUR_JUDGE_MODEL --base-url YOUR_JUDGE_BASE_URL --api-key-env YOUR_KEY_ENV --output experiments/dialogue_retest.json
```

Set the named environment variable to the judge API key before running. This script evaluates dialogue only and records individual responses, judge outputs, and errors. If `--source-report` is supplied without `--model-path`, the model path is taken from the report's `models.chat` field. To reevaluate a merged model, pass its path explicitly. Keep judge and prompt settings explicit when comparing results.

## Ablation and Plotting

Run a fixed-ratio SLERP ablation:

```bash
python code/ablation_merge_ratio.py --model-a /path/to/chat_model --model-b /path/to/base_model --output experiments/ratio_ablation
```

Run the separate layer-wise search workflow with coefficient visualization:

```bash
python code/layer_heatmap.py --model-a /path/to/chat_model --model-b /path/to/base_model --output experiments/layer_heatmap_run
```

Plot an existing per-layer search result:

```bash
python Figure_Code/plot_merge_result.py --output-dir experiments/layer_run
```

For paper figure inputs, dependencies, and plotting functions, see [Figure_Code/README.md](Figure_Code/README.md). For individual experiment scripts, see [code/README.md](code/README.md).

## Research Use

This project supports AI-safety research and alignment evaluation. Follow the licenses and terms of the models, datasets, and APIs used in an experiment, and do not use the software to facilitate real-world harm. See [LICENSE](LICENSE).
