from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_figures(raw_df: pd.DataFrame, overlap_df: pd.DataFrame, out_dir: Path, metric_col: str = "mape") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    _fig1(raw_df, out_dir / "fig1_mape_by_strategy.png", metric_col)
    _fig1_matrix(raw_df, out_dir / "fig1b_mape_by_strategy_model_matrix.png", metric_col)
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


def _fig1_matrix(raw_df: pd.DataFrame, path: Path, metric_col: str) -> None:
    datasets = sorted(raw_df["dataset"].unique())
    preferred_models = ["ridge", "extratrees", "xgboost"]
    present = set(raw_df["model"].unique())
    models = [m for m in preferred_models if m in present] + [m for m in sorted(present) if m not in preferred_models]
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(18, 14), sharey=False)

    for r, dataset in enumerate(datasets):
        row_data = raw_df[raw_df["dataset"] == dataset]
        row_agg = row_data.groupby(["model", "k_pct", "strategy"], as_index=False).agg(
            mean_metric=(metric_col, "mean"),
            std_metric=(metric_col, "std"),
        )
        if not row_agg.empty:
            row_mean = np.asarray(row_agg["mean_metric"], dtype=float)
            row_std = np.nan_to_num(np.asarray(row_agg["std_metric"], dtype=float), nan=0.0)
            row_y_top = float((row_mean + row_std).max())
            row_y_bottom = float((row_mean - row_std).min())
        else:
            row_y_top = 1.0
            row_y_bottom = 0.0
        if row_y_bottom > 0:
            row_y_bottom = 0.0
        if row_y_top <= row_y_bottom:
            row_y_top = row_y_bottom + 1.0

        for c, model in enumerate(models):
            ax = axes[r, c] if len(datasets) > 1 else axes[c]
            cell = raw_df[(raw_df["dataset"] == dataset) & (raw_df["model"] == model)]
            if cell.empty:
                ax.set_title(f"{dataset} | {model} (no data)")
                ax.axis("off")
                continue

            agg = cell.groupby(["k_pct", "strategy"], as_index=False).agg(
                mean_metric=(metric_col, "mean"),
                std_metric=(metric_col, "std"),
            )
            agg["k_label"] = (agg["k_pct"] * 100).astype(int).astype(str)
            sns.barplot(data=agg, x="k_label", y="mean_metric", hue="strategy", ax=ax, errorbar=None)

            for container, (_, g) in zip(ax.containers, agg.groupby("strategy"), strict=False):
                for bar, (_, row) in zip(container, g.iterrows(), strict=False):
                    ax.errorbar(
                        bar.get_x() + bar.get_width() / 2,
                        row["mean_metric"],
                        yerr=row["std_metric"],
                        fmt="none",
                        c="black",
                        capsize=2,
                    )

            ax.set_title(f"{dataset} | {model}")
            ax.set_xlabel("k%")
            ax.set_ylabel(f"Mean {metric_col.upper()}")
            ax.set_ylim(row_y_bottom, row_y_top)
            if r == 0 and c == len(models) - 1:
                ax.legend(title="strategy", loc="upper right")
            else:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

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
    required = {"dataset", "jaccard"}
    if overlap_df.empty or not required.issubset(set(overlap_df.columns)):
        ax.text(0.5, 0.5, "No SHAP/Tree overlap data available", ha="center", va="center")
        ax.set_title("SHAP vs Tree Top-50% Overlap")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Jaccard Similarity")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        sns.boxplot(data=overlap_df, x="dataset", y="jaccard", ax=ax)
        ax.set_title("SHAP vs Tree Top-50% Overlap")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Jaccard Similarity")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
