# Results Draft (Full Run)

## Experimental Scope

- Datasets: Superconductor, Communities and Crime, Bike Sharing.
- Models: Ridge, ExtraTrees, XGBoost.
- Feature-selection strategies: MI, RFE, Tree importance, SHAP.
- Feature-retention levels: 25%, 50%, 75%, 100%.
- Repeats/folds: 5 repeat seeds x 10 folds = 50 observations per cell.
- Total observations in `results_raw.csv`: 7,200.

## Data Completeness and Reproducibility

- `results_raw.csv` contains all expected combinations (balanced by dataset/model/strategy/k/fold/repeat).
- Per-dataset rows: 2,400 each.
- All configured repeat seeds and fold IDs are present.
- Timing metrics were collected per observation: selection, train, predict, and total wall-clock time.

## Metric Interpretation

- **Primary interpretive metric for model comparison in this draft: RMSE**.
- MAPE is retained, but should be interpreted cautiously for Communities and Crime because of zero/near-zero targets, which inflate relative errors.
- MSE and RMSE are therefore more stable for cross-dataset narrative.

## Baseline Definition

- Baseline is `k_pct = 1.0` (100% features, no feature removal).
- For each `(dataset, model, fold, repeat)`, baseline predictions are computed once and reused across strategy labels.
- Therefore, comparisons `x_vs_baseline100` quantify the effect of **feature removal + strategy**, not model changes.

## Overall Ranking Across All Cells (by mean RMSE rank)

- `tree`: 2.21 (best average rank)
- `shap`: 2.29
- `rfe`: 2.46
- `mi`: 3.04

Interpretation: Tree and SHAP are generally strongest overall, with MI weakest on average.

## Best Strategy per Dataset/Model at Each k (RMSE)

### Bike Sharing

- **ExtraTrees**: tree best at 25% and 50%; RFE best at 75%; baseline best at 100%.
- **Ridge**: RFE best at 25/50/75%; baseline best at 100%.
- **XGBoost**: tree best at 25%; SHAP best at 50% and 75%; baseline best at 100%.

### Communities and Crime

- **ExtraTrees**: tree best at 25% and 50%; MI best at 75% and 100%.
- **Ridge**: tree best at 25%; RFE best at 50% and 75%; baseline (MI-equivalent) best at 100%.
- **XGBoost**: MI best at 25% and 100%; tree best at 50%; SHAP best at 75%.

### Superconductor

- **ExtraTrees**: tree best at 25%; RFE best at 50%; MI best at 75% and 100%.
- **Ridge**: RFE best at 25/50/75%; baseline best at 100%.
- **XGBoost**: SHAP best at 25% and 50%; tree best at 75%; baseline best at 100%.

Interpretation: no single selector dominates all settings. SHAP is strongest in several XGBoost reduced-feature regimes, but not universally best.

## Penalty of Feature Reduction vs Baseline (Mean Delta RMSE %, k<100)

Selected highlights:

- **Bike Sharing / ExtraTrees**: tree +63.77%, shap +72.44%, rfe +101.68%, mi +130.95%.
- **Bike Sharing / XGBoost**: tree +46.19%, shap +51.73%, rfe +78.37%, mi +102.11%.
- **Superconductor / XGBoost**: shap +1.55%, tree +1.69%, rfe +3.52%, mi +5.56%.
- **Communities and Crime**: all methods are close to baseline (mostly +0.0% to +2.2%).

Interpretation:

- Bike Sharing is highly sensitive to aggressive feature removal.
- Communities and Crime is comparatively robust to feature subset changes (in RMSE terms).
- On Superconductor with XGBoost, SHAP and tree are closest to baseline under compression.

## Computational Cost

Average selection and total times (k<100):

- `rfe`: selection 3.61s, total 6.09s
- `mi`: selection 4.59s, total 6.81s
- `tree`: selection 6.80s, total 9.35s
- `shap`: selection 91.41s, total 93.78s

Interpretation: SHAP incurs a very large computational premium. Any SHAP performance gain should be discussed together with this cost.

## Statistical Testing (Wilcoxon + BH-FDR)

- Total tests: 180
- Significant after correction: 94

Significant counts by comparison:

- `mi_vs_baseline100`: 22
- `rfe_vs_baseline100`: 20
- `tree_vs_baseline100`: 20
- `shap_vs_baseline100`: 21
- `shap_vs_tree`: 11

Interpretation:

- Many reduced-feature settings are statistically different from baseline.
- SHAP vs Tree is significant in a subset of settings, indicating measurable but context-dependent differences.

## Selection Stability (Jaccard to Majority Set)

Mean stability over k<100:

- `mi`: 0.947
- `tree`: 0.917
- `rfe`: 0.910
- `shap`: 0.663

Interpretation: SHAP-selected subsets are materially less stable across repeats/folds than MI/Tree/RFE, despite occasional predictive advantages.

## Thesis-Ready Narrative

1. Baseline (all features) remains strongest in many cells, especially on Bike Sharing.
2. Under forced feature reduction, Tree and SHAP are most competitive overall, with SHAP particularly strong for several XGBoost settings.
3. SHAP's computational overhead is very high and its feature-subset stability is lower.
4. Therefore, method choice should be positioned as a trade-off:
   - **Best efficiency/stability**: Tree (and often RFE).
   - **Potential best reduced-feature accuracy in some regimes**: SHAP.

## Suggested Next Reporting Step

- Add direct pairwise tests at fixed k (`shap_vs_mi`, `shap_vs_rfe`, `shap_vs_tree`) per `(dataset, model, k)` to answer the core question: 
  "If features must be removed, which strategy is least harmful (or best)?"
