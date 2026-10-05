# EMMA Figure Code

Python plotting sources for Figures 5–13 of the EMMA paper.

## Files

| Figures | Content | Source |
|---|---|---|
| 5 | Global versus layer-wise search | `figure05_search_comparison.py`, `figure05_plot_helpers.py` |
| 6, 8 | Model comparison and dataset-level ASR | `figures06_08_model_and_dataset_comparison.py` |
| 7, 9, 10, 12 | Mixing trade-off, layer coefficients, formal convergence, and layer-segment statistics | `figures07_09_10_12_tradeoff_and_layers.py` |
| 11 | Test-search convergence | `figure11_test_convergence.py` |
| 13 | Run-log search convergence | `figure13_runlog_convergence.py` |
| Search diagnostics | Layer coefficients and convergence from an existing search run | `plot_merge_result.py` |

## Dependencies and Inputs

Install the plotting dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

The scripts use experiment reports, search histories, and logs:

- Figure 5: `coefficient_evidence.json` and the search-comparison plotting helper.
- Figures 6 and 8: `Figure06_data_audit.csv` and `Table12_dataset_asr_merged_review_v1.csv`.
- Figures 7, 9, 10, and 12: ablation reports, layer-search results, and formal search logs.
- Figure 11: test-run `history.json` files.
- Figure 13: formal search `run.log` files for the three model settings.

Supply the required inputs and set data/output paths before plotting. The scripts retain paths from the original project; they do not automatically resolve all inputs from the repository layout. Times New Roman is used for figure typography and must be available locally.

## Plotting Entry Points

For Figures 6 and 8, use `setup_matplotlib()` followed by `build_figure6()` or `build_figure8()`. The combined script's full `main()` also depends on historical manuscript and figure-building files.

For Figures 7, 9, 10, and 12, the corresponding functions are `fig5_ablation_tradeoff()`, `fig4_layer_coefficients_overview()`, `figA4_formal_layer_convergence()`, and `figA5_layer_segment_stats()`. Initialize the style and load the required records first; function names retain historical numbering.

For Figure 5, update the helper-file reference to `figure05_plot_helpers.py` and configure the coefficient-evidence path. The other dedicated scripts expose `main()` entry points once their input and output paths are configured.

## Search Result Plotting

To visualize an existing per-layer search result, run this command from the repository's `main/` directory:

```bash
python Figure_Code/plot_merge_result.py --output-dir experiments/per_layer_run
```

The script reads `best_so_far.json` and available convergence records from the specified directory. Use `--save` to specify the output image path.
