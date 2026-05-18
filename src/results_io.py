from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_COLUMNS = [
    "dataset",
    "model",
    "strategy",
    "k_pct",
    "fold",
    "repeat_seed",
    "mape",
    "mse",
    "rmse",
    "selection_time_s",
    "train_time_s",
    "predict_time_s",
    "total_time_s",
]


def load_raw_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.read_csv(path)
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[RAW_COLUMNS]


def save_raw_results(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby(["dataset", "model", "strategy", "k_pct"], as_index=False).agg(
        mean_mape=("mape", "mean"),
        mean_mse=("mse", "mean"),
        std_mape=("mape", "std"),
        median_mape=("mape", "median"),
        mean_rmse=("rmse", "mean"),
        mean_selection_time_s=("selection_time_s", "mean"),
        mean_train_time_s=("train_time_s", "mean"),
        mean_predict_time_s=("predict_time_s", "mean"),
        mean_total_time_s=("total_time_s", "mean"),
    )
    return grouped


def save_summary(path: Path, summary_df: pd.DataFrame) -> None:
    summary_df.to_csv(path, index=False)
