from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import CategoricalDtype
from sklearn.datasets import fetch_openml


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
REPORT_PATH = ROOT / "report.md"


@dataclass
class DatasetBundle:
    key: str
    name: str
    X_raw: pd.DataFrame
    y_raw: pd.Series
    X_proc: pd.DataFrame
    y_proc: pd.Series
    target_name: str


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _fetch(name: str, version: int | str) -> tuple[pd.DataFrame, pd.Series]:
    ds = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    X = ds.data.copy()
    y = ds.target.copy()
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    return X, y


def _encode_mixed_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if isinstance(s.dtype, CategoricalDtype):
            out[col] = s.cat.codes.astype(float)
            out.loc[out[col] < 0, col] = np.nan
        elif pd.api.types.is_object_dtype(s):
            out[col] = pd.factorize(s, sort=True)[0].astype(float)
            out.loc[out[col] < 0, col] = np.nan
        else:
            out[col] = pd.to_numeric(s, errors="coerce")
    return out


def load_superconductor() -> DatasetBundle:
    X, y = _fetch("superconduct", 1)
    y_num = pd.to_numeric(y, errors="coerce")
    X_num = _encode_mixed_numeric(X)
    df = X_num.copy()
    df["critical_temp"] = y_num
    df = df.dropna()
    y_proc = df.pop("critical_temp")
    X_proc = df
    return DatasetBundle("superconductor", "Superconductor", X, y_num, X_proc, y_proc, "critical_temp")


def load_communities_crime() -> DatasetBundle:
    X, y = _fetch("communities_and_crime", "active")
    y_num = pd.to_numeric(y, errors="coerce")
    X_num = _encode_mixed_numeric(X)
    keep = X_num.columns[X_num.isna().mean() <= 0.20]
    X_num = X_num[keep]
    df = X_num.copy()
    df["ViolentCrimesPerPop"] = y_num
    df = df.dropna()
    y_proc = df.pop("ViolentCrimesPerPop")
    X_proc = df
    return DatasetBundle("communities_crime", "Communities and Crime", X, y_num, X_proc, y_proc, "ViolentCrimesPerPop")


def load_bike_sharing() -> DatasetBundle:
    X, y = _fetch("bike_sharing_demand", 2)
    y_num = pd.to_numeric(y, errors="coerce")
    X2 = X.copy()
    for col in ["casual", "registered", "dteday"]:
        if col in X2.columns:
            X2 = X2.drop(columns=[col])
    X_num = _encode_mixed_numeric(X2)
    df = X_num.copy()
    df["cnt"] = y_num
    df = df.dropna()
    y_proc = df.pop("cnt")
    X_proc = df
    return DatasetBundle("bike_sharing", "Bike Sharing", X, y_num, X_proc, y_proc, "cnt")


def save_csv_snapshots(bundle: DatasetBundle) -> None:
    raw_df = bundle.X_raw.copy()
    raw_df[bundle.target_name] = bundle.y_raw
    raw_df.to_csv(RAW_DIR / f"{bundle.key}_raw.csv", index=False)

    proc_df = bundle.X_proc.copy()
    proc_df[bundle.target_name] = bundle.y_proc
    proc_df.to_csv(PROC_DIR / f"{bundle.key}_processed.csv", index=False)


def make_tables(bundle: DatasetBundle) -> dict[str, Path]:
    files: dict[str, Path] = {}
    df_proc = bundle.X_proc.copy()
    y = bundle.y_proc

    overview = pd.DataFrame(
        [
            {
                "dataset": bundle.name,
                "rows_raw": int(bundle.X_raw.shape[0]),
                "features_raw": int(bundle.X_raw.shape[1]),
                "rows_processed": int(df_proc.shape[0]),
                "features_processed": int(df_proc.shape[1]),
                "rows_removed": int(bundle.X_raw.shape[0] - df_proc.shape[0]),
            }
        ]
    )
    p = TABLES_DIR / f"{bundle.key}_overview.csv"
    overview.to_csv(p, index=False)
    files["overview"] = p

    missing = bundle.X_raw.isna().mean().sort_values(ascending=False).rename("missing_ratio").reset_index()
    missing = missing.rename(columns={"index": "feature"})
    p = TABLES_DIR / f"{bundle.key}_missingness.csv"
    missing.to_csv(p, index=False)
    files["missingness"] = p

    desc = df_proc.describe().T
    p = TABLES_DIR / f"{bundle.key}_feature_stats.csv"
    desc.to_csv(p)
    files["feature_stats"] = p

    target_stats = y.describe().to_frame(name="value")
    p = TABLES_DIR / f"{bundle.key}_target_stats.csv"
    target_stats.to_csv(p)
    files["target_stats"] = p

    corr = pd.concat([df_proc, y.rename(bundle.target_name)], axis=1).corr(numeric_only=True)
    p = TABLES_DIR / f"{bundle.key}_correlation_matrix.csv"
    corr.to_csv(p)
    files["correlation_matrix"] = p

    top_target_corr = corr[bundle.target_name].drop(bundle.target_name).abs().sort_values(ascending=False).head(20)
    p = TABLES_DIR / f"{bundle.key}_top_target_correlations.csv"
    top_target_corr.rename("abs_corr_with_target").to_csv(p)
    files["top_target_correlations"] = p
    return files


def make_figures(bundle: DatasetBundle) -> dict[str, Path]:
    files: dict[str, Path] = {}
    sns.set_theme(style="whitegrid")

    corr = pd.concat([bundle.X_proc, bundle.y_proc.rename(bundle.target_name)], axis=1).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title(f"{bundle.name} Correlation Matrix")
    p = FIG_DIR / f"{bundle.key}_correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    files["corr_heatmap"] = p

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(bundle.y_proc, bins=40, kde=True, ax=ax)
    ax.set_title(f"{bundle.name} Target Distribution ({bundle.target_name})")
    p = FIG_DIR / f"{bundle.key}_target_distribution.png"
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    files["target_distribution"] = p
    return files


def build_report(bundles: list[DatasetBundle], table_map: dict[str, dict[str, Path]], fig_map: dict[str, dict[str, Path]]) -> None:
    lines = [
        "# Pre-study Report",
        "",
        "This report documents source download, preprocessing, local dataset snapshots, and exploratory summaries.",
        "",
        "## Pipeline",
        "",
        "download from source -> preprocess -> save as csv -> report -> use on study",
        "",
    ]

    for b in bundles:
        lines.extend(
            [
                f"## {b.name}",
                "",
                f"- Source key: `{b.key}`",
                f"- Raw shape: `{b.X_raw.shape[0]} x {b.X_raw.shape[1]}`",
                f"- Processed shape: `{b.X_proc.shape[0]} x {b.X_proc.shape[1]}`",
                f"- Target: `{b.target_name}`",
                "",
                "### Saved data",
                "",
                f"- `{(RAW_DIR / f'{b.key}_raw.csv').relative_to(ROOT)}`",
                f"- `{(PROC_DIR / f'{b.key}_processed.csv').relative_to(ROOT)}`",
                "",
                "### Tables",
                "",
            ]
        )
        for _, p in table_map[b.key].items():
            lines.append(f"- `{p.relative_to(ROOT)}`")
        lines.extend(["", "### Figures", ""])
        for _, p in fig_map[b.key].items():
            lines.append(f"- `{p.relative_to(ROOT)}`")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    bundles = [load_superconductor(), load_communities_crime(), load_bike_sharing()]
    table_map: dict[str, dict[str, Path]] = {}
    fig_map: dict[str, dict[str, Path]] = {}

    for bundle in bundles:
        print(f"pre-study: {bundle.name}")
        save_csv_snapshots(bundle)
        table_map[bundle.key] = make_tables(bundle)
        fig_map[bundle.key] = make_figures(bundle)

    build_report(bundles, table_map, fig_map)
    print(f"report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
