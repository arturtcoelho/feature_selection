# Feature Selection Thesis Pipeline

This repository contains a reproducible experiment pipeline for comparing feature-selection strategies on regression tasks, plus a pre-study data-report pipeline.

## What this project does

- Downloads 3 datasets programmatically (OpenML/UCI mirrors).
- Cleans and preprocesses data.
- Runs cross-validated experiments for 3 models x 4 selection strategies x multiple feature-retention levels.
- Stores raw fold-level results, summary tables, statistical tests, stability analysis, figures, selected feature sets, and feature-importance rankings.

## Project structure

- `main.py`: single entry point (CLI).
- `src/`: experiment modules (data loading, feature selection, runner, stats, figures, stability).
- `pre_study/`: independent data pre-study pipeline and report generation.
- `outputs/`: default output location when no run id is used.
- `runs/<run-id>/...`: isolated output location for safe/non-overwriting runs.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Pre-study pipeline (recommended first)

Build reproducible local data snapshots and exploratory report:

```bash
python3 pre_study/build_pre_study.py
```

Generated artifacts:

- `pre_study/data/raw/*.csv`
- `pre_study/data/processed/*.csv`
- `pre_study/tables/*.csv`
- `pre_study/figures/*.png`
- `pre_study/report.md`

## 2) Experiment pipeline

### Safe run pattern (recommended)

Use `--run-id` so outputs are isolated and never mixed:

```bash
python3 main.py --step all --run-id run01 --use-preprocessed-dir pre_study/data/processed
```

This writes to:

- `runs/run01/outputs/`
- `runs/run01/figures/`
- `runs/run01/logs/`

If `run01` already exists, the command fails unless `--resume` is provided.

### Continue an interrupted run

```bash
python3 main.py --step all --run-id run01 --resume --use-preprocessed-dir pre_study/data/processed
```

### Quick smoke run

```bash
python3 main.py --quick --step all --run-id quick01 --use-preprocessed-dir pre_study/data/processed
```

## Main CLI options

- `--step {all,experiments,summary,stats,stability,figures}`
- `--quick` (2 folds x 2 repeats x k=[50,100])
- `--resume` (skip already-computed rows)
- `--run-id <id>` (safe isolated run folders)
- `--use-preprocessed-dir <dir>` (use local processed CSV snapshots)
- `--metric {rmse,mse,mape}` (used in stats + figures; default `rmse`)
- `--workers <n>` (process count for model-level parallelism)
- `--no-gpu` (force CPU)
- `--corr-prune-threshold <float>` (optional rank-then-prune correlation filter, in `(0,1)`)

## Output files

Under either `outputs/` (default) or `runs/<run-id>/outputs/`:

- `results_raw.csv`: fold-level results with all metrics/timings.
- `results_summary.csv`: aggregated summary statistics.
- `stats_tests.csv`: Wilcoxon + BH-FDR results.
- `stability_analysis.csv`: Jaccard stability by strategy and k.
- `selections_raw.csv`: selected features per fold/repeat.
- `feature_importances_raw.csv`: full ranked feature lists and scores per fold/repeat.

Figures are stored under `figures/` or `runs/<run-id>/figures/`.

## Baseline and strategy notes

- Baseline = `k_pct = 1.0` (no feature removal).
- For `k < 1.0`, ranking is computed on training fold only.
- `tree` strategy uses a separate `ExtraTreesRegressor` ranking step.
- `shap` strategy builds a SHAP ranking and reuses it for 25/50/75% subsets.
- Correlation pruning (if enabled) is applied **after ranking** and **before top-k selection**.

## Typical experiment workflow

1. Build pre-study data/report.
2. Run baseline experiment in isolated run folder.
3. Run variant with correlation pruning in another run folder.
4. Compare summaries/stats/figures across run folders.

Example:

```bash
python3 main.py --step all --run-id run_base --use-preprocessed-dir pre_study/data/processed
python3 main.py --step all --run-id run_corr09 --use-preprocessed-dir pre_study/data/processed --corr-prune-threshold 0.9
```
