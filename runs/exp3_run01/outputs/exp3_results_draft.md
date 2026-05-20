# Experiment 3 Results Draft

- Dataset: Allstate
- Model: XGBoost
- Raw observations: 800

## Best by k (RMSE)
|   k_pct | strategy                  |   mean_rmse |   mean_mae |   mean_mape |
|--------:|:--------------------------|------------:|-----------:|------------:|
|    0.05 | native_fi                 |     2198.79 |    1435.99 |    0.810259 |
|    0.1  | hybrid_fi_custom_shap_rfe |     2016.12 |    1292.59 |    0.693514 |
|    0.15 | hybrid_fi_custom_shap_rfe |     1932.24 |    1211.56 |    0.623415 |
|    0.25 | custom_shap_rfe           |     1925.7  |    1202.34 |    0.616579 |
|    0.5  | native_fi                 |     1912.92 |    1195.68 |    0.61256  |
|    1    | baseline                  |     1913.12 |    1195.69 |    0.612588 |