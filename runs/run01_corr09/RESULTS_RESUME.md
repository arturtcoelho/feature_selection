# Results Resume - `run01_corr09`

## Run Context

- Run ID: `run01_corr09`
- Correlation pruning: `corr_prune_threshold = 0.9`
- Protocol: 3 datasets x 3 models x 4 strategies x 4 k-levels x 10 folds x 5 repeats
- Primary interpretation metric in this resume: **RMSE**

## Artifacts Produced

- `runs/run01_corr09/outputs/results_raw.csv` (7200 rows)
- `runs/run01_corr09/outputs/results_summary.csv` (144 grouped rows)
- `runs/run01_corr09/outputs/stats_tests.csv` (180 tests)
- `runs/run01_corr09/outputs/stability_analysis.csv` (48 rows)
- `runs/run01_corr09/outputs/selections_raw.csv` (selected feature sets + pruning metadata)
- `runs/run01_corr09/outputs/feature_importances_raw.csv` (full ranking + scores per fold/repeat)

## What Correlation Pruning Did

Mean post-prune candidate feature counts (`n_features_post_prune`):

- Bike Sharing: `11.0` (from 12)
- Communities and Crime: `71.4` (range 69-74 from ~100)
- Superconductor: `45.9` (range 43-48 from 81)

Interpretation: pruning had strongest compression on Superconductor and Communities/Crime, and mild compression on Bike Sharing.

## Best Strategy by Dataset/Model/k (RMSE)

Highlights from the best-per-cell table:

- **Bike Sharing**
  - ExtraTrees: `tree` best at 25/50%; `rfe` best at 75%; baseline best at 100%
  - Ridge: `rfe` best at 25/50/75%; baseline best at 100%
  - XGBoost: `tree` best at 25/50/75%; baseline best at 100%
- **Communities and Crime**
  - ExtraTrees: `tree` best at 25/50%; `mi` best at 75/100%
  - Ridge: `rfe` best at 25/50/75%; baseline best at 100%
  - XGBoost: mixed (`mi` best at 25/75/100%, `shap` best at 50%)
- **Superconductor**
  - ExtraTrees: mixed (`rfe` at 25, `shap` at 50, `tree` at 75, baseline at 100)
  - Ridge: `rfe` at 25, `shap` at 50, `tree` at 75, baseline at 100
  - XGBoost: `shap` at 25, `tree` at 50/75, baseline at 100

Interpretation: no universal winner. With correlation pruning enabled, `tree` and `rfe` are frequently strongest; `shap` wins specific cells, especially on Superconductor/XGBoost low-k settings.

## Strategy-Level Aggregate View

Average across all grouped cells (`results_summary.csv`):

- `tree`: mean RMSE `32.77` (best), mean selection time `4.37s`
- `shap`: mean RMSE `33.18`, mean selection time `73.41s`
- `rfe`: mean RMSE `35.47`, mean selection time `0.75s`
- `mi`: mean RMSE `36.56`, mean selection time `4.26s`

Interpretation: `tree` gives the strongest global accuracy/cost balance in this run; `shap` is competitive in accuracy but much more expensive.

## Statistical Tests (Wilcoxon + BH-FDR)

- Total tests: `180`
- Significant after correction: `100`

Significant counts by comparison:

- `mi_vs_baseline100`: 24
- `rfe_vs_baseline100`: 20
- `tree_vs_baseline100`: 22
- `shap_vs_baseline100`: 23
- `shap_vs_tree`: 11

Interpretation: many reduced-feature conditions differ significantly from baseline; SHAP vs Tree differs significantly in a subset, confirming meaningful method differences in several contexts.

## Stability of Selected Features

Mean Jaccard-to-majority for `k < 1.0`:

- `rfe`: `0.915`
- `mi`: `0.913`
- `tree`: `0.868`
- `shap`: `0.665`

Interpretation: SHAP remains clearly less stable in selected subsets across folds/repeats, while RFE/MI are most stable.

## Concrete Example of Persisted Importance Data

For Superconductor / XGBoost / SHAP / k=50%, frequently selected features include:

- `range_ThermalConductivity`
- `wtd_std_ElectronAffinity`
- `wtd_mean_ThermalConductivity`
- `mean_Density`
- `wtd_std_Valence`

These frequencies are traceable directly via `feature_importances_raw.csv` + `selections_raw.csv`, so you can analyze redundancy and ranking behavior without rerunning experiments.

## Practical Takeaways

1. Correlation pruning successfully reduces redundant candidate sets, especially for high-dimensional datasets.
2. Best predictive strategy is dataset/model/k dependent; no single method dominates everywhere.
3. `tree` is the strongest overall compromise in this run (accuracy + runtime).
4. `shap` can be best in specific cells but remains expensive and less stable.
5. Baseline (`k=100%`) is still hard to beat in many settings, so reduced-feature gains are selective rather than universal.
