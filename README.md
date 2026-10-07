# HealthFactor-AI

A research pipeline for drinking-water safety in İzmir, combining **current-state classification** with **future safety-trend prediction**. The repository contains data-refresh and cleaning scripts, a shared Health Factor calculation, model comparisons and reviewer-facing outputs.

[Case study](https://onurerguden.dev/en/projects/water-safety) · [Research context and current status](https://onurerguden.dev/en/research) · [Onur Ergüden](https://onurerguden.dev)

## How it works

```text
İZSU analysis data → cleaning → shared HF/WAWQI calculation → feature tables
                                                        ├─ reactive classification
                                                        ├─ proactive trend classification
                                                        └─ next-observation HF regression
```

- **Reactive layer:** classifies the current state as Good, Caution or Risk.
- **Proactive layer:** models changes in water-safety trends.
- **Regression:** predicts the Health Factor at the next observation.
- **Evaluation:** separates real, synthetic-only and mixed test results; model code includes temporal splitting and leakage-audit outputs.

## Where to start

| Path | Purpose |
| --- | --- |
| `data/update_data.py` | Incremental data refresh from İZSU |
| `data/clean_izsu.py` | Cleaning and normalization |
| `data/build_hf_and_features.py` | Health Factor and feature construction |
| `utils/` | Shared parameter definitions and HF calculation |
| `models/classification_model.py` | Reactive experiments and classwise evaluation |
| `models/proactive_trend_model.py` | Proactive model comparisons |
| `models/hf_next_observation_regression.py` | Next-observation regression |
| `models/run_all_models.py` | Sequential model-suite runner with logs and a run manifest |
| `models/outputs/` | Committed model outputs |
| `paper_revision_deliverables/` | Research revision snapshot and supporting artifacts |

## Run the model suite

Use a Python environment with the dependencies imported by the selected scripts. The repository currently has no consolidated dependency lockfile; model comparisons use pandas, NumPy, scikit-learn, matplotlib, joblib and XGBoost, with additional dependencies for report generation.

From the repository root, with dependencies installed:

```sh
python models/run_all_models.py \
  --input data/data/izsu_features.csv \
  --output-root outputs
```

This reads the committed feature snapshot and writes fresh outputs, logs and `run_all_manifest.json` under `outputs/`. It does not refresh the source dataset. To select individual layers, use `--skip-reactive`, `--skip-proactive` or `--skip-regression`. Inspect all runner options with:

```sh
python models/run_all_models.py --help
```

## Interpreting results

The real dataset may not contain enough examples of every risk class. The reactive code explicitly marks non-estimable single-class experiments as `SKIPPED_SINGLE_CLASS` and evaluates controlled synthetic scenarios separately. Synthetic contamination results should not be read as field performance on observed toxic water.

Inspect the [reactive metrics](models/outputs/reactive/reactive_model_metrics.csv), [proactive metrics](models/outputs/proactive/proactive_model_metrics.csv), split summaries and leakage audits together. Historical outputs in `models/legacy/` belong to earlier experiments; compare results only after checking their data, target and split definitions.

## Research

The project supports *Two-layered Artificial Intelligence System to Assess and Forecast the Safety Level of Drinking Water Resources*, co-authored by O. Ergüden, B. Ceylani, A. Şengül, S. Yılmaz, E. Yılmaz, M. Y. Kalkan and D. E. Fawzy. See the [portfolio research page](https://onurerguden.dev/en/research) for current publication status and context.

This repository is a research implementation, not an operational certification of drinking-water safety.
