from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml


def _encode_mixed_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_categorical_dtype(series):
            out[col] = series.cat.codes.astype(float)
            out.loc[out[col] < 0, col] = pd.NA
        elif pd.api.types.is_object_dtype(series):
            out[col] = pd.factorize(series, sort=True)[0].astype(float)
            out.loc[out[col] < 0, col] = pd.NA
        else:
            out[col] = pd.to_numeric(series, errors="coerce")
    return out


def _fetch_openml_candidates(candidates: list[tuple[str, int | str | None]]) -> tuple[pd.DataFrame, pd.Series]:
    for name, version in candidates:
        try:
            kwargs = {"name": name, "as_frame": True, "parser": "auto"}
            if version is not None:
                kwargs["version"] = version
            ds: Any = fetch_openml(**kwargs)
            X = ds.data.copy()
            y = ds.target.copy()
            if isinstance(X, pd.Series):
                X = X.to_frame()
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            return X, y
        except Exception as exc:  # noqa: BLE001
            logging.warning("OpenML fetch failed for %s v%s: %s", name, version, exc)
    raise RuntimeError(f"Could not fetch any dataset candidate: {candidates}")


def load_superconductor() -> tuple[pd.DataFrame, pd.Series, str]:
    X, y = _fetch_openml_candidates(
        [
            ("superconduct", 1),
            ("Superconduct", 1),
        ]
    )
    y = pd.to_numeric(y, errors="coerce")
    df = X.copy()
    df["target"] = y
    df = df.dropna()
    y_clean = df.pop("target")
    X_clean = df.apply(pd.to_numeric, errors="coerce").dropna()
    if isinstance(X_clean, pd.Series):
        X_clean = X_clean.to_frame()
    y_clean = y_clean.loc[X_clean.index]
    return X_clean, y_clean, "Superconductor"


def load_communities_crime() -> tuple[pd.DataFrame, pd.Series, str]:
    X, y = _fetch_openml_candidates(
        [
            ("communities_and_crime", "active"),
            ("Communities_and_Crime", "active"),
        ]
    )
    y = pd.to_numeric(y, errors="coerce")
    X_num = X.apply(pd.to_numeric, errors="coerce")
    missing_ratio = X_num.isna().mean()
    X_num = X_num.loc[:, missing_ratio <= 0.20]
    df = X_num.copy()
    df["target"] = y
    df = df.dropna()
    y_clean = df.pop("target")
    X_clean = df
    if isinstance(X_clean, pd.Series):
        X_clean = X_clean.to_frame()
    return X_clean, y_clean, "Communities and Crime"


def load_bike_sharing() -> tuple[pd.DataFrame, pd.Series, str]:
    X, y = _fetch_openml_candidates(
        [
            ("bike_sharing_demand", 2),
            ("Bike_Sharing_Demand", 2),
            ("bike_sharing", "active"),
        ]
    )
    df = X.copy()

    target_col = None
    for candidate in ["cnt", "count", "target"]:
        if candidate in df.columns:
            target_col = candidate
            break

    if y is not None and target_col is None:
        target_series = pd.to_numeric(y, errors="coerce")
    elif target_col is not None:
        target_series = pd.to_numeric(df.pop(target_col), errors="coerce")
    else:
        raise RuntimeError("Could not determine Bike Sharing target column")

    for leak_col in ["casual", "registered", "dteday"]:
        if leak_col in df.columns:
            df = df.drop(columns=[leak_col])

    X_num = _encode_mixed_numeric(df)
    clean_df = X_num.copy()
    clean_df["target"] = target_series
    clean_df = clean_df.dropna()
    y_clean = clean_df.pop("target")
    X_clean = clean_df
    if isinstance(X_clean, pd.Series):
        X_clean = X_clean.to_frame()
    return X_clean, y_clean, "Bike Sharing"


def load_all_datasets() -> list[dict]:
    loaders = [load_superconductor, load_communities_crime, load_bike_sharing]
    datasets = []
    for loader in loaders:
        X, y, name = loader()
        datasets.append(
            {
                "name": name,
                "X": X,
                "y": y,
                "feature_names": list(X.columns),
            }
        )
        logging.info("Loaded %s: rows=%d, features=%d", name, X.shape[0], X.shape[1])
    return datasets


def load_all_datasets_from_local(processed_dir: str) -> list[dict]:
    files = [
        ("superconductor_processed.csv", "Superconductor", "critical_temp"),
        ("communities_crime_processed.csv", "Communities and Crime", "ViolentCrimesPerPop"),
        ("bike_sharing_processed.csv", "Bike Sharing", "cnt"),
    ]
    datasets = []
    for file_name, ds_name, target_col in files:
        path = f"{processed_dir}/{file_name}"
        df = pd.read_csv(path)
        if target_col not in df.columns:
            raise RuntimeError(f"Target column {target_col} missing in {path}")
        y = pd.to_numeric(df[target_col], errors="coerce")
        X = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")
        clean = X.copy()
        clean["target"] = y
        clean = clean.dropna()
        y_clean = clean.pop("target")
        X_clean = clean
        datasets.append(
            {
                "name": ds_name,
                "X": X_clean,
                "y": y_clean,
                "feature_names": list(X_clean.columns),
            }
        )
        logging.info("Loaded local %s: rows=%d, features=%d", ds_name, X_clean.shape[0], X_clean.shape[1])
    return datasets
