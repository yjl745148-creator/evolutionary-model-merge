# EMMA Figure Code

Plot Figures 5–13 directly from the experiment records in `../date/`.

## Install and Run

Run these commands from the repository's `main/` directory:

```bash
python -m pip install -r Figure_Code/requirements.txt
python Figure_Code/figure05_search_comparison.py
python Figure_Code/figures06_08_model_and_dataset_comparison.py
python Figure_Code/figures07_09_10_12_tradeoff_and_layers.py
python Figure_Code/figure11_test_convergence.py
python Figure_Code/figure13_runlog_convergence.py
```

These commands write PNG, PDF, and SVG figures to `main/outputs/`. Paths are resolved relative to the scripts, so input/output locations do not depend on the working directory. Generated outputs are ignored by Git.

## Files

| Figures | Content | Source |
|---|---|---|
| 5 | Global versus layer-wise search | `figure05_search_comparison.py`, `figure05_plot_helpers.py` |
| 6, 8 | Model comparison and dataset-level ASR | `figures06_08_model_and_dataset_comparison.py` |
| 7, 9, 10, 12 | Mixing trade-off, layer coefficients, formal convergence, and layer-segment statistics | `figures07_09_10_12_tradeoff_and_layers.py` |
| 11 | Test-search convergence | `figure11_test_convergence.py` |
| 13 | Run-log search convergence | `figure13_runlog_convergence.py` |
| Shared utilities | Repository paths, fonts, and formal-report readers | `figure_common.py` |
| Search diagnostics | Plot an individual search output directory | `plot_merge_result.py` |

## Data Sources

- Figure 5 reads the 8B global/layer `results.yaml` files and comparison reports. Layer coefficients are decoded with sigmoid and averaged; the global scalar is decoded with sigmoid. Both DQR values use the global report's chat-model reference, as in the paper.
- Figures 6 and 8 read formal comparison reports directly. The 8B merged result uses the layer-search report, while its chat/base baselines use the global-search report. The 3B and SmolLM2 comparisons use each model's layer-search report. ASR uses valid per-item judge scores with a threshold of 7. Composite is `0.4 * dialogue + 0.6 * (10 * ASR)`, with ASR expressed as a fraction.
- Figures 7, 9, 10, and 12 use ablation JSON files, formal layer-search YAML results, and logs.
- Figure 11 uses `date/*/layer_search_test/**/history.json`.
- Figure 13 uses the three formal layer-search logs and also writes `figure13_runlog_data.csv`.

The plotting scripts do not run model inference or call judge APIs. Times New Roman is selected when installed; otherwise Matplotlib's DejaVu Serif is used. Font substitution can change text spacing.

## Individual Search Results

To plot an existing per-layer search output, run from `main/`:

```bash
python Figure_Code/plot_merge_result.py --output-dir experiments/per_layer_run
```

This separate diagnostic tool reads `best_so_far.json` and available convergence records. Use `--save` to choose its image output path.
