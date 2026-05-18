from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _to_set(features_str: str) -> set[str]:
    if not features_str:
        return set()
    return set(features_str.split("|"))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def compute_stability_analysis(selection_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, strategy, k_pct), group in selection_df.groupby(["dataset", "strategy", "k_pct"]):
        sets = [_to_set(s) for s in group["selected_features"].tolist()]
        if not sets:
            continue

        all_features = sorted(set().union(*sets))
        votes = {f: 0 for f in all_features}
        for s in sets:
            for f in s:
                votes[f] += 1
        threshold = len(sets) / 2
        majority = {f for f, cnt in votes.items() if cnt >= threshold}
        jac = [jaccard(s, majority) for s in sets]
        rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "k_pct": k_pct,
                "mean_jaccard_to_majority": float(np.mean(jac)),
            }
        )
    return pd.DataFrame(rows)


def compute_shap_tree_overlap(selection_df: pd.DataFrame, k_pct: float = 0.5) -> pd.DataFrame:
    rows = []
    for dataset, ds_group in selection_df[selection_df["k_pct"] == k_pct].groupby("dataset"):
        grp_shap = ds_group[ds_group["strategy"] == "shap"]
        grp_tree = ds_group[ds_group["strategy"] == "tree"]
        merged = grp_shap.merge(
            grp_tree,
            on=["dataset", "model", "k_pct", "fold", "repeat_seed"],
            suffixes=("_shap", "_tree"),
        )
        for _, row in merged.iterrows():
            jac = jaccard(_to_set(row["selected_features_shap"]), _to_set(row["selected_features_tree"]))
            rows.append(
                {
                    "dataset": dataset,
                    "model": row["model"],
                    "fold": row["fold"],
                    "repeat_seed": row["repeat_seed"],
                    "jaccard": jac,
                }
            )
    return pd.DataFrame(rows)
