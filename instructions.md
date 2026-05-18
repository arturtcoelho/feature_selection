## Goal
Build a complete, reproducible ML experiment pipeline comparing four feature selection strategies for regression tasks across three datasets and three models. All code in a single Python script (or Jupyter notebook). Results saved to CSV and PNG figures. The output should be directly usable in an academic paper at masters level.

---

## Datasets
Download programmatically from UCI / OpenML. Do NOT use local files.

1. **Superconductor** (UCI): ~21k rows, 81 features. Target: `critical_temp`.
2. **Communities and Crime** (UCI): ~1994 rows, ~100 features (drop non-numeric and columns with >20% missing). Target: `ViolentCrimesPerPop`.
3. **Bike Sharing** (UCI, hourly dataset): ~17k rows, 16 features. Target: `cnt`. Drop `casual` and `registered` (they are components of `cnt`, would cause leakage). Drop `dteday`.

For each dataset:
- Drop rows with any remaining NaN after initial cleaning
- Standardise all features with StandardScaler (fit on train, apply to test)
- Keep a fixed random seed (SEED = 42) everywhere

---

## Models
Implement three regressors:
1. `Ridge` (alpha=1.0) — sklearn
2. `ExtraTreesRegressor` (n_estimators=200, random_state=SEED)
3. `XGBRegressor` (n_estimators=200, learning_rate=0.05, random_state=SEED, verbosity=0)

---

## Feature Selection Strategies
Implement four strategies. Each strategy must return a ranked list of features (most to least important) so we can apply top-k% cutoffs consistently.

1. **Mutual Information (MI)** — `mutual_info_regression` from sklearn. Fit on training split only. Rank by score descending.

2. **Recursive Feature Elimination (RFE)** — `RFE`. Rank features by `ranking_` attribute (lower = better selected; invert for descending rank).

3. **Tree Gini/Gain Importance** — Rank by `feature_importances_` descending.

4. **SHAP** — Compute SHAP values with `shap.TreeExplainer`, rank by mean(|SHAP values|) descending across all training samples.

Important: all four selection strategies are computed ONLY on the training fold, never on the test fold.

---

## Experimental Protocol

### Cross-validation
- Use **10-fold cross-validation** with `KFold(n_splits=10, shuffle=True, random_state=SEED)`
- For each fold: fit selection strategy on train split → select features → fit model on reduced train → evaluate on test split

### Feature retention levels (k)
Test four levels: **25%, 50%, 75%, 100%** of available features.
- At 100%, skip feature selection entirely (baseline — all features, all strategies return the same result; compute once and reuse)
- Always keep at least 1 feature (use `max(1, int(round(n_features * k)))`)

### Metric
Primary metric: **MAPE** (Mean Absolute Percentage Error).
- Use `mean_absolute_percentage_error` from sklearn
- Also record **RMSE** as a secondary metric (for the appendix)
- Also record **wall-clock time** for the feature selection step alone (use `time.perf_counter()`), separately from model training time

### Repeat for stability
Run the entire 10-fold CV **5 times** with different random seeds for the KFold: seeds = [42, 123, 256, 512, 999].
This gives 10 folds × 5 repeats = 50 MAPE observations per (dataset, model, strategy, k) combination.

---

## Output: Results Table
Save a CSV `results_raw.csv` with one row per observation:

| dataset | model | strategy | k_pct | fold | repeat_seed | mape | rmse | selection_time_s |

Then compute an aggregated summary `results_summary.csv`:
- Group by (dataset, model, strategy, k_pct)
- Report: mean_mape, std_mape, median_mape, mean_rmse, mean_selection_time_s

---

## Statistical Tests
After collecting all results, run a **Wilcoxon signed-rank test** for each (dataset, model, k_pct) combination:
- Compare each of {MI, RFE, Tree, SHAP} against the **baseline (100%, same model)** — 4 tests
- Also compare SHAP vs Tree importance directly — 1 test
- Correct for multiple comparisons with **Benjamini-Hochberg FDR** (use `statsmodels.stats.multitest.multipletests`)
- Save results to `stats_tests.csv` with columns: dataset, model, k_pct, comparison, statistic, p_value, p_value_corrected, significant (bool, alpha=0.05)

---

## Figures to Generate
Save all figures as high-res PNG (300 dpi) in a `/figures` subfolder.

**Figure 1 — MAPE by strategy and k% (main result)**
For each dataset (3 subplots), show a grouped bar chart:
- x-axis: k% levels (25, 50, 75, 100)
- y-axis: mean MAPE averaged across all 3 models and 50 repeats
- bars grouped by strategy (4 colors)
- error bars = ±1 std
- title per subplot = dataset name

**Figure 2 — MAPE heatmap (dataset × strategy, at k=50%)**
For each model (3 heatmaps side by side):
- rows = datasets, columns = strategies
- cell = mean MAPE (annotated)
- use a single consistent color scale per figure

**Figure 3 — Computational cost**
For each dataset (3 subplots):
- x-axis: strategy (MI, RFE, Tree, SHAP)
- y-axis: mean selection time in seconds (log scale)
- bar chart, aggregated across all k% and models

**Figure 4 — Stability: MAPE variance by strategy**
For each dataset:
- boxplot of MAPE distributions across all 50 repeats
- one box per (strategy, k%) combination — use k=50% only to keep it readable
- grouped by strategy on x-axis

**Figure 5 — SHAP vs Tree importance feature overlap**
For each dataset, compute the top-k selected features (at k=50%) for Tree and SHAP, across all 50 repeats:
- compute Jaccard similarity between the two feature sets per repeat
- plot distribution as a boxplot per dataset

---

## Stability Analysis Table
Save `stability_analysis.csv`:
- For each (dataset, strategy, k_pct): compute the mean Jaccard similarity of the selected feature sets across the 50 repeats (comparing each repeat's feature set to the majority-vote feature set)
- This measures how consistent each strategy's selection is

---

## Code Structure Requirements
- Single self-contained script `experiments.py` (or notebook `experiments.ipynb`)
- Use a `tqdm` progress bar wrapping the outer loops
- Use comprehensive logging for all steps, be verbose
- Print a summary table to stdout at the end using `tabulate`
- Reproducibility: set numpy, random, and sklearn random states everywhere
- Handle edge cases: if MAPE is inf (target contains zeros or near-zeros), skip that fold and log a warning; do not crash
- Add a `--quick` CLI flag that runs only 2 folds × 2 repeats × k=[50,100] for fast testing

---

## Dependencies
Use only: numpy, pandas, scikit-learn, xgboost, shap, matplotlib, seaborn, statsmodels, tqdm, tabulate. All installable via pip.

---

## Final Deliverables Checklist
- [ ] `experiments.py` — full pipeline
- [ ] `results_raw.csv` — all 50 observations per cell
- [ ] `results_summary.csv` — aggregated stats
- [ ] `stats_tests.csv` — Wilcoxon + BH results
- [ ] `stability_analysis.csv` — Jaccard consistency
- [ ] `/figures/fig1_mape_by_strategy.png`
- [ ] `/figures/fig2_mape_heatmap.png`
- [ ] `/figures/fig3_compute_cost.png`
- [ ] `/figures/fig4_mape_stability.png`
- [ ] `/figures/fig5_shap_tree_overlap.png`

