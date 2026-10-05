# EMMA Figure Code

Plotting source code for Figures 5–13 of the EMMA paper.

| Figure | Source |
|---|---|
| 5: Global vs. layer-wise search | `figure05_search_comparison.py` + `figure05_plot_helpers.py` |
| 6: Model comparison; 8: Dataset-level ASR | `figures06_08_model_and_dataset_comparison.py` (`build_figure6`, `build_figure8`) |
| 7: Mixing trade-off; 9: Layer coefficients; 10: Formal convergence; 12: Layer statistics | `figures07_09_10_12_tradeoff_and_layers.py` |
| 11: Test convergence | `figure11_test_convergence.py` |
| 13: Run-log convergence | `figure13_runlog_convergence.py` |

Dependencies are listed in `requirements.txt`. Experiment data and fonts are not included. These are collected plotting sources, not a standalone reproduction package: original data/output paths and historical figure labels require adjustment before running. Figure 5 also needs `coefficient_evidence.json`; Figures 6/8 need their summary CSVs; the remaining figures use experiment JSON/YAML/log files. Do not run the Figures 6/8 script's full `main()`; it also invokes unrelated historical figure builders.
