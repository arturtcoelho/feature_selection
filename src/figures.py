from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_figures(raw_df: pd.DataFrame, overlap_df: pd.DataFrame, out_dir: Path, metric_col: str = "mape") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    _fig1(raw_df, out_dir / "fig1_mape_by_strategy.png", metric_col)
    _fig2(raw_df, out_dir / "fig2_mape_heatmap.png", metric_col)
    _fig3(raw_df, out_dir / "fig3_compute_cost.png")
    _fig4(raw_df, out_dir / "fig4_mape_stability.png", metric_col)
    _fig5(overlap_df, out_dir / "fig5_shap_tree_overlap.png")


def _fig1(raw_df: pd.DataFrame, path: Path, metric_col: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, (dataset, df_ds) in zip(axes, raw_df.groupby("dataset"), strict=False):
        agg = df_ds.groupby(["k_pct", "strategy"], as_index=False).agg(mean_metric=(metric_col, "mean"), std_metric=(metric_col, "std"))
        agg["k_label"] = (agg["k_pct"] * 100).astype(int).astype(str)
        sns.barplot(data=agg, x="k_label", y="mean_metric", hue="strategy", ax=ax, errorbar=None)
        for container, (_, g) in zip(ax.containers, agg.groupby("strategy"), strict=False):
            for bar, (_, row) in zip(container, g.iterrows(), strict=False):
                ax.errorbar(bar.get_x() + bar.get_width() / 2, row["mean_metric"], yerr=row["std_metric"], fmt="none", c="black", capsize=2)
        ax.set_title(dataset)
        ax.set_xlabel("k%")
        ax.set_ylabel(f"Mean {metric_col.upper()}")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _fig2(raw_df: pd.DataFrame, path: Path, metric_col: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    models = sorted(raw_df["model"].unique())
    for ax, model in zip(axes, models, strict=False):
        df_m = raw_df[(raw_df["model"] == model) & (raw_df["k_pct"] == 0.5)]
        piv = df_m.pivot_table(index="dataset", columns="strategy", values=metric_col, aggfunc="mean")
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title(model)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _fig3(raw_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (dataset, df_ds) in zip(axes, raw_df.groupby("dataset"), strict=False):
        agg = df_ds[df_ds["k_pct"] < 1.0].groupby("strategy", as_index=False).agg(mean_selection_time_s=("selection_time_s", "mean"))
        sns.barplot(data=agg, x="strategy", y="mean_selection_time_s", ax=ax)
        ax.set_yscale("log")
        ax.set_title(dataset)
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Selection Time (s, log)")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _fig4(raw_df: pd.DataFrame, path: Path, metric_col: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    df = raw_df[raw_df["k_pct"] == 0.5].copy()
    for ax, (dataset, df_ds) in zip(axes, df.groupby("dataset"), strict=False):
        sns.boxplot(data=df_ds, x="strategy", y=metric_col, ax=ax)
        ax.set_title(dataset)
        ax.set_xlabel("Strategy")
        ax.set_ylabel(metric_col.upper())
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _fig5(overlap_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=overlap_df, x="dataset", y="jaccard", ax=ax)
    ax.set_title("SHAP vs Tree Top-50% Overlap")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Jaccard Similarity")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
