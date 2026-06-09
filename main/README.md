# Evolutionary Model Merge

**English** | [简体中文](README.zh-CN.md)

This project uses CMA-ES to search merge parameters for two same-architecture
Hugging Face causal language models, and ships scripts for ratio ablation,
per-layer search, model comparison, and result plotting.

> ⚠️ **Responsible use**: This project includes safety / red-teaming evaluation
> capabilities (Attack Success Rate, ASR) that probe how a model behaves under
> harmful requests. It is intended solely for AI-safety research, alignment
> evaluation, and defensive purposes. See [Responsible Use](#responsible-use) below.

## Requirements

Python 3.10 or 3.11 is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

Model merging generally needs a CUDA GPU. CPU works for some checks, but full
experiments are slow.

Optional LLM-judge API configuration:

```bash
# Qwen / DashScope
export LLM_PROVIDER=qwen
export QWEN_API_KEY="..."

# OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY="..."
```

If the corresponding API key is unset, the merge stage warns explicitly and
falls back to rule-based scoring only.

## Main entry point

Global merge:

```bash
python run_merge.py \
  --model-a /path/to/chat-model \
  --model-b /path/to/base-model \
  --output experiments/example \
  --seed 42
```

Low-VRAM mode:

```bash
python run_merge.py \
  --model-a /path/to/chat-model \
  --model-b /path/to/base-model \
  --output experiments/example \
  --low-vram
```

Per-layer search:

```bash
python run_merge.py \
  --model-a /path/to/chat-model \
  --model-b /path/to/base-model \
  --output experiments/per-layer \
  --per-layer-search \
  --per-layer-alpha 0.7 \
  --large-model never
```

`--per-layer-alpha` controls the risk weight in the per-layer objective:

```text
alpha * (1 - safety_score) + (1 - alpha) * dialogue_score
```

## Resume from checkpoint

```bash
python run_merge.py \
  --model-a /path/to/chat-model \
  --model-b /path/to/base-model \
  --output experiments/example \
  --resume
```

Resume supports:

- two-model global scalar search
- DataFlow routing parameters
- DARE drop rate and its random seed
- per-layer search parameters

The CLI/YAML config used to resume must match the one that produced the checkpoint.

## Data directory

`compare_models.py` looks for the following under the project's `data/eval/`:

```text
data/eval/
  mt_bench_questions.jsonl
  alpaca_eval.json
  advbench_behaviors.jsonl
  harmbench_behaviors.jsonl
  jailbreakbench_behaviors.jsonl
```

If a file is missing, the script falls back to a small built-in prompt set.
These harmful-evaluation datasets are **not distributed with this repository**;
obtain them from their official sources (see [Dataset sources](#dataset-sources))
and comply with each dataset's license.

## Model comparison

```bash
python compare_models.py \
  --merged experiments/example/merged_model \
  --chat /path/to/chat-model \
  --base /path/to/base-model \
  --output experiments/example/comparison_report.json
```

**Prompt policy**: during evaluation all three models (merged/chat/base) apply
their own tokenizer chat template; models without a template (typically the base
model) automatically fall back to a raw prompt. Evaluation injects **no system
prompt** (including no safety prompt). Empty responses are excluded and flagged
rather than scored as 0.

If all judge requests fail, the script raises an error instead of recording an
ASR of 0 from zero valid samples.

## Other scripts

Ratio ablation:

```bash
python ablation_merge_ratio.py \
  --model-a /path/to/chat-model \
  --model-b /path/to/base-model \
  --output experiments/ablation
```

Standalone per-layer heatmap search:

```bash
python layer_heatmap.py \
  --model-a /path/to/chat-model \
  --model-b /path/to/base-model \
  --output experiments/layer-heatmap
```

Re-plot per-layer results:

```bash
python plot_merge_result.py \
  --output-dir experiments/per-layer
```

Dialogue-only retest (DeepSeek judge):

`retest_it_dialogue_deepseek.py` re-evaluates **only the dialogue score**
(MT-Bench + AlpacaEval). It skips the safety/ASR evaluation and uses an
independent DeepSeek V4 Pro judge.

**Why it exists:** in some runs the dialogue scores from the main
`compare_models.py` come out as an all-zero column — for example the model under
test returns empty responses under the given prompt format, or the primary judge
call fails — which makes that model's dialogue ability invalid/underestimated.
Rather than rerunning the full comparison, use this script to re-test just the
dialogue score for the affected model (e.g. the merged / m2 model) and cross-check
it with a separate judge. Every item is recorded (input prompt, model response,
raw judge output, parsed score, and any error) for debugging.

```bash
# Option 1: point at the model directly
python retest_it_dialogue_deepseek.py \
  --model-path /path/to/m2_model \
  --model-id m2 \
  --output experiments/example/m2_dialogue_retest.json

# Option 2: inherit model path / sample counts / prompt mode from an existing report
python retest_it_dialogue_deepseek.py \
  --source-report experiments/example/comparison_report.json \
  --output experiments/example/m2_dialogue_retest.json
```

The judge API key is read from an environment variable (use `--api-key-env` to
choose which one); the judge model defaults to `deepseek-v4-pro` (override with
`--judge-model` / `--base-url`).

## Output files

A typical merge run produces:

```text
experiments/example/
  best_so_far.json
  latest_generation_best.json
  generation_best.jsonl
  results.yaml
  merged_model/
```

Each new (non-resume) run re-initializes this run's `generation_best.jsonl` and
the global-best record, so old experiments do not contaminate a new run.
`--resume` does not re-run CMA-ES.

The three CMA stages use `seed`, `seed + 1`, and `seed + 2` respectively, so the
candidate sequence is reproducible under the same config and environment.

## Dataset sources

The evaluation datasets below are **not included in this repository**. Download
them from their official sources, place them under `data/eval/`, and comply with
each one's license / terms of use:

- **MT-Bench** — https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
- **AlpacaEval** — https://github.com/tatsu-lab/alpaca_eval
- **AdvBench** — https://github.com/llm-attacks/llm-attacks
- **HarmBench** — https://github.com/centerforaisafety/HarmBench
- **JailbreakBench** — https://github.com/JailbreakBench/jailbreakbench

The last three are harmful-behavior prompt datasets used only for safety evaluation.

## Responsible use

This project is intended for **AI-safety research and alignment evaluation**. Its
Attack Success Rate (ASR) evaluation sends harmful requests to the model under
test and scores the responses in order to **measure and improve** model safety —
not to generate or spread harmful content.

By using this software you agree to:

- use it only for lawful research, evaluation, and defensive purposes;
- not generate, distribute, or operationalize content or actions that cause
  real-world harm;
- comply with the licenses of the models, datasets, and APIs you use, and with
  applicable laws and regulations;
- take full responsibility for any harmful datasets you obtain and place under
  `data/eval/`.

The authors accept no liability for misuse of this software. See [LICENSE](LICENSE).

## Notes

- The two source models must have compatible layer count, hidden size, attention
  heads, vocabulary, and embedding configuration.
- `--trust-remote-code` is off by default; enable it only for trusted models.
- On receiving a termination signal, the program stops after finishing the
  current candidate evaluation and does not enter later CMA stages.


