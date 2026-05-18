from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def _safe_wilcoxon(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    diff = (a - b).to_numpy()
    if len(diff) < 2:
        return np.nan, 1.0
    if np.allclose(diff, 0.0, rtol=0.0, atol=1e-15):
        return 0.0, 1.0
    try:
        stat, pval = wilcoxon(a, b, zero_method="wilcox")
        if np.isnan(pval):
            return float(stat), 1.0
        return float(stat), float(pval)
    except ValueError:
        return np.nan, 1.0


def run_statistical_tests(raw_df: pd.DataFrame, alpha: float = 0.05, metric_col: str = "mape") -> pd.DataFrame:
    rows = []
    grouped = raw_df.groupby(["dataset", "model", "k_pct"], as_index=False)

    for _, group in grouped:
        dataset = group["dataset"].iloc[0]
        model = group["model"].iloc[0]
        k_pct = group["k_pct"].iloc[0]

        baseline = raw_df[
            (raw_df["dataset"] == dataset)
            & (raw_df["model"] == model)
            & (raw_df["strategy"] == "tree")
            & (raw_df["k_pct"] == 1.0)
        ][["repeat_seed", "fold", metric_col]].rename(columns={metric_col: "metric_base"})

        comparisons = ["mi", "rfe", "tree", "shap"]
        for strategy in comparisons:
            target = raw_df[
                (raw_df["dataset"] == dataset)
                & (raw_df["model"] == model)
                & (raw_df["strategy"] == strategy)
                & (raw_df["k_pct"] == k_pct)
            ][["repeat_seed", "fold", metric_col]].rename(columns={metric_col: "metric_target"})
            merged = baseline.merge(target, on=["repeat_seed", "fold"], how="inner")
            if len(merged) < 2:
                continue
            stat, pval = _safe_wilcoxon(merged["metric_target"], merged["metric_base"])
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "k_pct": k_pct,
                    "comparison": f"{strategy}_vs_baseline100",
                    "statistic": stat,
                    "p_value": pval,
                }
            )

        shap_df = raw_df[
            (raw_df["dataset"] == dataset)
            & (raw_df["model"] == model)
            & (raw_df["strategy"] == "shap")
            & (raw_df["k_pct"] == k_pct)
        ][["repeat_seed", "fold", metric_col]].rename(columns={metric_col: "metric_shap"})
        tree_df = raw_df[
            (raw_df["dataset"] == dataset)
            & (raw_df["model"] == model)
            & (raw_df["strategy"] == "tree")
            & (raw_df["k_pct"] == k_pct)
        ][["repeat_seed", "fold", metric_col]].rename(columns={metric_col: "metric_tree"})
        merged_st = shap_df.merge(tree_df, on=["repeat_seed", "fold"], how="inner")
        if len(merged_st) >= 2:
            stat, pval = _safe_wilcoxon(merged_st["metric_shap"], merged_st["metric_tree"])
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "k_pct": k_pct,
                    "comparison": "shap_vs_tree",
                    "statistic": stat,
                    "p_value": pval,
                }
            )

    tests_df = pd.DataFrame(rows)
    if tests_df.empty:
        tests_df["p_value_corrected"] = []
        tests_df["significant"] = []
        return tests_df

    reject, p_corr, _, _ = multipletests(tests_df["p_value"].fillna(1.0), alpha=alpha, method="fdr_bh")
    tests_df["p_value_corrected"] = p_corr
    tests_df["significant"] = reject
    return tests_df
