# EMMA Experiment Records

Stored experiment reports, search results, and logs for the LLaMA3-3B, LLaMA3-8B, and SmolLM2 settings. The directory name `date/` is retained because the plotting scripts use this path.

## Directory Structure

| Model directory | Main subdirectories |
|---|---|
| [llama3b/](llama3b/) | `global_search/`, `layer_search/`, `layer_search_test/1–3/`, `ablation/1–3/` |
| [llama8b/](llama8b/) | `global_search/`, `layer_search/`, `layer_search_test/`, `ablation/1–3/` |
| [smollm2/](smollm2/) | `compare/global/`, `compare/layer/`, `layer_search_test/`, `layer_heatmap/`, `ablation/1–3/` |

For LLaMA models, formal layer-search records are under `layer_search/layer_merge/`; comparison reports are under `layer_search/compare_layer_merge/`. SmolLM2 formal layer-search records and evaluation reports are stored together in `compare/layer/`.

## File Types

| File | Contents |
|---|---|
| `comparison_report.json`, `llama3_8b_2_report.json` | Formal evaluation summaries and per-item records. |
| `results.yaml` | Saved search configuration, optimization history, and selected solution. |
| `run.log`, `compare.log` | Search or evaluation execution logs. |
| `history.json` | Candidate-level records from test searches. |
| `result.json` | Test-search settings and selected coefficients; available for some runs. |
| `ablation_results.json` | Evaluation results across fixed mixing ratios. |
| `it_dialogue_template_full.json` | Separate dialogue-only reevaluation, including its judge, prompt settings, and item-level results. |
| `layer_heatmap.png` | Stored visualization of a layer-search result. |

The `system_prompt/` folders contain separate merged-model diagnostic reports. Keep them distinct from the `no_system_prompt/` comparisons. Test-search histories and formal search results are also separate record types.

## Paper Figure Inputs

| Figures | Records used |
|---|---|
| 5 | 8B global/layer search YAML files and evaluation reports. |
| 6, 8 | Formal model comparison reports and their per-item ASR records. |
| 7 | `ablation/*/ablation_results.json` for each model. |
| 9, 12 | Formal layer-search coefficients from `results.yaml`. |
| 10, 13 | Formal layer-search logs; the Figure 10 loader also supports generation summaries from YAML. |
| 11 | `layer_search_test/**/history.json`. |

For Figures 6 and 8, the 8B merged result comes from the layer-search report, while its chat/base baselines come from the global-search report. The 3B and SmolLM2 comparisons use their respective layer-search reports. The separate dialogue-only reevaluation is not substituted into these figures. Judge, sample-count, and prompt settings can differ across runs; consult each report's metadata when interpreting comparisons.

## Usage

See [Figure_Code/README.md](../Figure_Code/README.md) for plotting commands. The scripts read these records and write generated figures to `main/outputs/` without modifying the inputs.

These are experiment records, not model checkpoints or a complete benchmark installation. Evaluation prompt files used by `code/compare_models.py` belong in `main/code/data/eval/`; see the [project README](../README.md) for setup and dataset sources.
