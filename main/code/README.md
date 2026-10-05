# EMMA Code

Implementation of evolutionary model merging, evaluation, and experiment analysis.

## Files

| File | Purpose |
|---|---|
| `run_merge.py` | Command-line entry point for configuring and running evolutionary merging. |
| `evolutionary_model_merge.py` | Core merging implementation, CMA-ES search, and proxy evaluators. |
| `compare_models.py` | Safety and dialogue evaluation of merged, chat, and base models. |
| `ablation_merge_ratio.py` | Fixed-ratio SLERP sweep with evaluation across mixing coefficients. |
| `layer_heatmap.py` | Per-layer CMA-ES search with fast/full evaluation and coefficient visualization. |
| `plot_merge_result.py` | Plot coefficients and convergence from an existing search output directory. |
| `retest_it_dialogue_deepseek.py` | Dialogue-only reevaluation of a chat/instruct model with a configurable API judge. |

## Setup

Run these commands from the repository's `main/` directory:

```bash
python -m pip install -r requirements.txt
python code/run_merge.py --help
python code/compare_models.py --help
```

Provide compatible chat/instruct and base checkpoints, the required evaluation data, and API credentials for the selected judge. Set model paths, sampling options, and output locations for each experiment. Keep the scripts together because several import sibling modules.

## Examples

Start a merge search:

```bash
python code/run_merge.py --model-a /path/to/chat_model --model-b /path/to/base_model --output experiments/merge_run
```

Run a fixed-ratio ablation:

```bash
python code/ablation_merge_ratio.py --model-a /path/to/chat_model --model-b /path/to/base_model --output experiments/ratio_ablation
```

Plot an existing per-layer search result:

```bash
python code/plot_merge_result.py --output-dir experiments/per_layer_run
```

The plotting command reads `best_so_far.json` and available convergence records. Use each script's `--help` for its options. Paper figure sources are maintained separately in `../Figure_Code/`.
