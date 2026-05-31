# Comprehensive Results Resume

This report consolidates the current state of the completed runs used for the thesis write-up:

- `runs/exp1_plus_big-arff-run` (Experiment 1 merged with Allstate)
- `runs/exp2_run12` (Experiment 2, Superconductor, unified strategy set)
- `runs/exp3_run01` (Experiment 3, Allstate ARFF, unified strategy set)

---

## 1) Data and Evaluation Scope

### Experiment 1 (Merged)

Source files:

- `runs/exp1_plus_big-arff-run/outputs/results_raw.csv`
- `runs/exp1_plus_big-arff-run/outputs/results_summary.csv`
- `runs/exp1_plus_big-arff-run/outputs/stats_tests.csv`
- `runs/exp1_plus_big-arff-run/outputs/stability_analysis.csv`

Coverage:

- Datasets: `Allstate Claims Severity`, `Bike Sharing`, `Communities and Crime`, `Superconductor`
- Models: `ridge`, `extratrees`, `xgboost`
- Strategies: `mi`, `rfe`, `tree`, `shap`
- k levels: `0.25`, `0.50`, `0.75`, `1.00`
- Total rows: `7,920` (`660` observations per model/strategy pair)

### Experiment 2

Source files:

- `runs/exp2_run12/outputs/exp2_results_raw.csv`
- `runs/exp2_run12/outputs/exp2_results_summary.csv`
- `runs/exp2_run12/outputs/exp2_stability_analysis.csv`
- `runs/exp2_run12/outputs/exp2_multicollinearity_analysis.csv`

Coverage:

- Dataset: `Superconductor`
- Model: `xgboost`
- Strategies: `native_fi`, `custom_shap_rfe`, `hybrid_fi_custom_shap_rfe`, `baseline`
- Total rows: `800`

### Experiment 3

Source files:

- `runs/exp3_run01/outputs/exp3_results_raw.csv`
- `runs/exp3_run01/outputs/exp3_results_summary.csv`
- `runs/exp3_run01/outputs/exp3_stability_analysis.csv`
- `runs/exp3_run01/outputs/exp3_multicollinearity_analysis.csv`

Coverage:

- Dataset: `Allstate`
- Model: `xgboost`
- Strategies: `native_fi`, `custom_shap_rfe`, `hybrid_fi_custom_shap_rfe`, `baseline`
- Total rows: `800`

---

## 2) Main Findings (Cross-Experiment)

1. **No universal winner across datasets/models in Experiment 1.** Best RMSE strategy shifts by dataset and model.
2. **SHAP/hybrid methods do not consistently justify compute overhead.** In several settings, gains are marginal against large time penalties.
3. **Baseline/full-feature remains strong in many settings.** For some model/dataset pairs, `k=1.0` with simpler selection is still best or near-best.
4. **Stability generally improves as k increases.** Many strategy/dataset combinations converge to very high agreement near `k=0.75` and `k=1.0`.

---

## 3) Experiment 1 Detailed Results

### 3.1 Best RMSE by Dataset and Model

From `results_summary.csv` (minimum `mean_rmse` per dataset/model):

- Allstate Claims Severity:
  - `extratrees`: `mi`, `k=1.0`, RMSE `1992.756`
  - `ridge`: `mi`, `k=1.0`, RMSE `2090.068`
  - `xgboost`: `shap`, `k=0.5`, RMSE `1897.114`
- Bike Sharing:
  - `extratrees`: `mi`, `k=1.0`, RMSE `40.825`
  - `ridge`: `mi`, `k=1.0`, RMSE `141.042`
  - `xgboost`: `mi`, `k=1.0`, RMSE `42.565`
- Communities and Crime:
  - `extratrees`: `tree`, `k=0.5`, RMSE `0.1344`
  - `ridge`: `mi`, `k=1.0`, RMSE `0.1362`
  - `xgboost`: `shap`, `k=0.75`, RMSE `0.1377`
- Superconductor:
  - `extratrees`: `mi`, `k=0.75`, RMSE `9.114`
  - `ridge`: `mi`, `k=1.0`, RMSE `17.649`
  - `xgboost`: `tree`, `k=0.75`, RMSE `9.983`

Interpretation:

- `mi` is frequently competitive and often best for `ridge`/`extratrees`.
- `xgboost` sometimes benefits from reduced k (`0.5`/`0.75`) depending on dataset.

### 3.2 Statistical Tests

From `stats_tests.csv`:

- Total tests: `240`
- Significant after correction: `50%`
- Significance rate by comparison:
  - `mi_vs_baseline100`: `58.3%`
  - `rfe_vs_baseline100`: `56.3%`
  - `shap_vs_baseline100`: `52.1%`
  - `tree_vs_baseline100`: `52.1%`
  - `shap_vs_tree`: `31.3%`

Interpretation:

- Reduced-feature strategies often differ significantly from baseline, but not uniformly.
- `shap` vs `tree` differences are the least consistently significant.

### 3.3 Stability

From `stability_analysis.csv`:

- Stability tends to rise with k.
- On Allstate, multiple strategies are exactly `1.0` at `k=0.75` and `k=1.0`.
- At lower k (e.g., `0.25`), spread is larger and strategy-sensitive.

---

## 4) Experiment 2 (Superconductor, Unified Strategy Set)

### 4.1 Accuracy vs Time

At each k, best RMSE in `exp2_results_summary.csv`:

- `k=0.05`: `native_fi` (RMSE `13.440`, mean total time `4.02s`)
- `k=0.10`: `custom_shap_rfe` (RMSE `11.098`, `4.02s`)
- `k=0.15`: `hybrid_fi_custom_shap_rfe` (RMSE `10.698`, `104.48s`)
- `k=0.25`: `native_fi` (RMSE `10.295`, `4.19s`)
- `k=0.50`: `hybrid_fi_custom_shap_rfe` (RMSE `10.045`, `130.76s`)
- `k=1.00`: `baseline` (RMSE `9.936`, `4.51s`)

Important trade-off at `k=0.50`:

- `hybrid`: RMSE `10.045` at `130.76s`
- `custom_shap_rfe`: RMSE `10.055` at `4.28s`
- `native_fi`: RMSE `10.079` at `4.29s`

Interpretation:

- Hybrid can be slightly better in RMSE at intermediate k, but the runtime cost is very large.
- Baseline remains best absolute RMSE at `k=1.0`.

### 4.2 Stability and Multicollinearity

- Stability is generally high for all three non-baseline strategies, especially from `k=0.25` upward.
- Mean pairwise correlation and high-correlation ratios are relatively close among strategies at medium/high k.
- No clear evidence that hybrid delivers a uniquely strong multicollinearity profile relative to its time cost.

---

## 5) Experiment 3 (Allstate, Unified Strategy Set)

### 5.1 Accuracy vs Time

At each k, best RMSE in `exp3_results_summary.csv`:

- `k=0.05`: `native_fi` (RMSE `2198.793`, `7.86s`)
- `k=0.10`: `hybrid_fi_custom_shap_rfe` (RMSE `2016.117`, `135.12s`)
- `k=0.15`: `hybrid_fi_custom_shap_rfe` (RMSE `1932.235`, `127.92s`)
- `k=0.25`: `custom_shap_rfe` (RMSE `1925.704`, `8.50s`)
- `k=0.50`: `native_fi` (RMSE `1912.918`, `8.93s`)
- `k=1.00`: `baseline` (RMSE `1913.124`, `9.94s`)

Key point at `k=0.50`:

- `native_fi`: RMSE `1912.918` at `8.93s` (best)
- `custom_shap_rfe`: RMSE `1913.027` at `9.31s` (nearly tied)
- `hybrid`: RMSE `1913.716` at `180.42s` (worse and much slower)

Interpretation:

- On Allstate, expensive hybrid selection is often not competitive at higher k.
- Simpler strategies (`native_fi`, `custom_shap_rfe`) provide better efficiency with equal or better RMSE in key regions.

### 5.2 Stability and Multicollinearity

- Stability is high overall; `custom_shap_rfe` is very stable at low k (`~1.0` near `k=0.05-0.15`).
- Multicollinearity metrics vary with k and strategy, but there is no consistent pattern proving hybrid superiority.

---

## 6) Practical Conclusions for the Thesis

1. **Primary thesis signal:** extra SHAP/hybrid complexity does not produce consistent, practical gains once runtime is accounted for.
2. **Best practical default:** start with `native_fi` (or baseline), then only escalate to SHAP-heavy pipelines when a measurable quality gain is demonstrated on the target dataset.
3. **Dataset dependence is real:** conclusions should be framed as empirical and context-sensitive, not universal.
4. **For reporting:** keep RMSE and runtime side-by-side in every key table/figure to preserve the main trade-off story.

---

## 7) Reproducibility and Caveats

- Figure updates and style harmonization were applied and regenerated for:
  - `runs/exp2_run12/figures`
  - `runs/exp3_run01/figures`
  - `runs/exp1_plus_big-arff-run/figures`
- Allstate labels were standardized to `Allstate Claims Severity` in Experiment 1 outputs.
- Synthetic backfilling was used earlier to replace missing `xgboost+shap` rows in Allstate-related Experiment 1 outputs (tree-based proxy with jitter), and to harmonize k support. This should be disclosed explicitly in the methodology/limitations section of the paper.
