# Experiment 2 Results Draft

- Dataset: Superconductor
- Model: XGBoost
- Raw observations: 900

## Best by k (RMSE)
|   k_pct | strategy                  |   mean_rmse |   mean_mae |   mean_mape |
|--------:|:--------------------------|------------:|-----------:|------------:|
|    0.05 | native_fi                 |    13.44    |    8.68867 |     8.25338 |
|    0.10 | custom_shap_rfe           |    11.0985  |    7.0121  |     7.61    |
|    0.15 | hybrid_fi_custom_shap_rfe |    10.6982  |    6.73444 |     7.24992 |
|    0.25 | native_fi                 |    10.2954  |    6.46234 |     7.51385 |
|    0.50 | hybrid_fi_custom_shap_rfe |    10.0453  |    6.25191 |     7.24023 |
