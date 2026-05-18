from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import SEED, paths
from src.data_loading import load_all_datasets_from_local
from src.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2: Superconductor + XGBoost feature-selection study")
    parser.add_argument(
        "--step",
        choices=["all", "experiments", "summary", "stability", "multicollinearity", "figures", "analysis"],
        default="all",
    )
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true", help="2 folds x 2 repeats x k=[0.5,1.0]")
    parser.add_argument("--use-preprocessed-dir", type=str, default="pre_study/data/processed")
    return parser.parse_args()


def make_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=seed,
        verbosity=0,
        n_jobs=-1,
        tree_method="hist",
    )


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = mean_absolute_percentage_error(y_true, y_pred)
    return float(m) if np.isfinite(m) else np.nan


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _fi_rank(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str], seed: int) -> list[str]:
    model = make_model(seed)
    model.fit(X_train, y_train)
    order = np.argsort(model.feature_importances_)[::-1]
    return [feature_names[i] for i in order]


def _shaprfecv_select(X_train_df: pd.DataFrame, y_train: np.ndarray, k_keep: int, seed: int) -> list[str]:
    from probatus.feature_elimination import ShapRFECV

    model = make_model(seed)
    selector = ShapRFECV(model=model, step=1, cv=3, scoring="neg_root_mean_squared_error", n_jobs=1)
    selector.fit_compute(X_train_df, y_train)
    try:
        selected = selector.get_reduced_features_set(num_features=k_keep)
    except TypeError:
        selected = selector.get_reduced_features_set(k_keep)
    selected = list(selected)
    if len(selected) < k_keep:
        missing = [c for c in X_train_df.columns if c not in selected]
        selected.extend(missing[: k_keep - len(selected)])
    return selected[:k_keep]


def run_experiments(p: dict[str, Path], quick: bool, resume: bool, preprocessed_dir: str) -> None:
    ds_all = load_all_datasets_from_local(preprocessed_dir)
    ds = [d for d in ds_all if d["name"] == "Superconductor"][0]
    X_df = ds["X"].copy()
    y = ds["y"].to_numpy()
    feature_names = list(X_df.columns)

    repeats = [42, 123] if quick else [42, 123, 256, 512, 999]
    folds = 2 if quick else 10
    k_levels = [0.5, 1.0] if quick else [0.25, 0.5, 0.75, 1.0]
    strategies = ["native_fi", "shaprfecv", "hybrid_fi_shaprfecv"]

    raw_path = p["outputs"] / "exp2_results_raw.csv"
    sel_path = p["outputs"] / "exp2_selections_raw.csv"
    imp_path = p["outputs"] / "exp2_importances_raw.csv"

    existing = pd.read_csv(raw_path) if (resume and raw_path.exists()) else pd.DataFrame()
    done = set()
    if not existing.empty:
        for _, r in existing.iterrows():
            done.add((r["strategy"], float(r["k_pct"]), int(r["fold"]), int(r["repeat_seed"])))

    rows: list[dict] = []
    sels: list[dict] = []
    imps: list[dict] = []

    for rep_i, seed in enumerate(repeats, 1):
        print(f"repeat {rep_i}/{len(repeats)} (seed={seed})")
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(kf.split(X_df), 1):
            print(f"fold {fold}/{folds}")
            X_tr_raw = X_df.iloc[tr].to_numpy()
            X_te_raw = X_df.iloc[te].to_numpy()
            y_tr = y[tr]
            y_te = y[te]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)
            X_tr_df = pd.DataFrame(X_tr, columns=feature_names)
            X_te_df = pd.DataFrame(X_te, columns=feature_names)
            n_feat = len(feature_names)

            # precompute rankings once per fold
            t0 = time.perf_counter()
            fi_rank = _fi_rank(X_tr, y_tr, feature_names, seed)
            fi_time = time.perf_counter() - t0
            imps.append({"strategy": "native_fi", "fold": fold, "repeat_seed": seed, "ranked_features": "|".join(fi_rank)})

            for strategy in strategies:
                for k in k_levels:
                    key = (strategy, float(k), fold, seed)
                    if key in done:
                        continue
                    k_keep = max(1, int(round(n_feat * k)))

                    sel_t0 = time.perf_counter()
                    if k == 1.0:
                        selected = feature_names.copy()
                        stage_fi_keep = feature_names.copy()
                        stage_shap_keep = feature_names.copy()
                    elif strategy == "native_fi":
                        selected = fi_rank[:k_keep]
                        stage_fi_keep = selected.copy()
                        stage_shap_keep = selected.copy()
                    elif strategy == "shaprfecv":
                        selected = _shaprfecv_select(X_tr_df, y_tr, k_keep, seed)
                        stage_fi_keep = feature_names.copy()
                        stage_shap_keep = selected.copy()
                    else:
                        # hybrid: drop more with FI first
                        n_drop = n_feat - k_keep
                        drop_fi = int(np.ceil(n_drop / 2.0))
                        keep_after_fi = [f for f in fi_rank if f not in set(fi_rank[-drop_fi:])]
                        inter_k = max(k_keep, len(keep_after_fi) - (n_drop - drop_fi))
                        inter_df = X_tr_df[keep_after_fi]
                        selected_inter = _shaprfecv_select(inter_df, y_tr, inter_k, seed)
                        selected = selected_inter[:k_keep]
                        stage_fi_keep = keep_after_fi
                        stage_shap_keep = selected_inter
                    selection_time_s = time.perf_counter() - sel_t0

                    sel_idx = [feature_names.index(c) for c in selected]
                    Xtr_sel = X_tr[:, sel_idx]
                    Xte_sel = X_te[:, sel_idx]

                    train_t0 = time.perf_counter()
                    model = make_model(seed)
                    model.fit(Xtr_sel, y_tr)
                    train_time_s = time.perf_counter() - train_t0
                    pred_t0 = time.perf_counter()
                    pred = model.predict(Xte_sel)
                    predict_time_s = time.perf_counter() - pred_t0

                    rmse = root_mean_squared_error(y_te, pred)
                    mae = mean_absolute_error(y_te, pred)
                    mape = _safe_mape(y_te, pred)
                    mse = mean_squared_error(y_te, pred)

                    rows.append(
                        {
                            "dataset": "Superconductor",
                            "model": "xgboost",
                            "strategy": strategy,
                            "k_pct": float(k),
                            "fold": fold,
                            "repeat_seed": seed,
                            "rmse": float(rmse),
                            "mae": float(mae),
                            "mape": float(mape),
                            "mse": float(mse),
                            "selection_time_s": float(selection_time_s),
                            "train_time_s": float(train_time_s),
                            "predict_time_s": float(predict_time_s),
                            "total_time_s": float(selection_time_s + train_time_s + predict_time_s),
                        }
                    )
                    sels.append(
                        {
                            "dataset": "Superconductor",
                            "model": "xgboost",
                            "strategy": strategy,
                            "k_pct": float(k),
                            "fold": fold,
                            "repeat_seed": seed,
                            "n_features_total": n_feat,
                            "n_features_selected": len(selected),
                            "n_features_after_fi_stage": len(stage_fi_keep),
                            "n_features_after_shap_stage": len(stage_shap_keep),
                            "selected_features_fi_stage": "|".join(stage_fi_keep),
                            "selected_features_shap_stage": "|".join(stage_shap_keep),
                            "selected_features": "|".join(selected),
                        }
                    )

    raw_df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if rows else existing
    raw_df.to_csv(raw_path, index=False)
    pd.DataFrame(sels).to_csv(sel_path, index=False)
    pd.DataFrame(imps).to_csv(imp_path, index=False)


def run_summary(p: dict[str, Path]) -> None:
    raw = pd.read_csv(p["outputs"] / "exp2_results_raw.csv")
    summary = raw.groupby(["dataset", "model", "strategy", "k_pct"], as_index=False).agg(
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"),
        mean_mape=("mape", "mean"),
        mean_selection_time_s=("selection_time_s", "mean"),
        mean_total_time_s=("total_time_s", "mean"),
    )
    summary.to_csv(p["outputs"] / "exp2_results_summary.csv", index=False)


def run_stability(p: dict[str, Path]) -> None:
    sel = pd.read_csv(p["outputs"] / "exp2_selections_raw.csv")
    rows = []
    for (strategy, k), g in sel.groupby(["strategy", "k_pct"]):
        sets = [set(x.split("|")) for x in g["selected_features"].tolist()]
        votes = {}
        for s in sets:
            for f in s:
                votes[f] = votes.get(f, 0) + 1
        majority = {f for f, c in votes.items() if c >= len(sets) / 2}
        vals = [_jaccard(s, majority) for s in sets]
        rows.append({"strategy": strategy, "k_pct": k, "mean_jaccard_to_majority": float(np.mean(vals))})
    pd.DataFrame(rows).to_csv(p["outputs"] / "exp2_stability_analysis.csv", index=False)


def run_multicollinearity(p: dict[str, Path], preprocessed_dir: str) -> None:
    ds_all = load_all_datasets_from_local(preprocessed_dir)
    ds = [d for d in ds_all if d["name"] == "Superconductor"][0]
    X_df = ds["X"]
    sel = pd.read_csv(p["outputs"] / "exp2_selections_raw.csv")
    out = []
    for (strategy, k), g in sel.groupby(["strategy", "k_pct"]):
        mean_abs_corr = []
        high_corr_ratio = []
        for feats in g["selected_features"]:
            cols = feats.split("|")
            c = X_df[cols].corr().abs().to_numpy()
            if c.shape[0] <= 1:
                mean_abs_corr.append(0.0)
                high_corr_ratio.append(0.0)
                continue
            tri = c[np.triu_indices(c.shape[0], k=1)]
            mean_abs_corr.append(float(np.mean(tri)))
            high_corr_ratio.append(float(np.mean(tri >= 0.8)))
        out.append(
            {
                "strategy": strategy,
                "k_pct": float(k),
                "mean_abs_pairwise_corr": float(np.mean(mean_abs_corr)),
                "mean_high_corr_pair_ratio_r_ge_0_8": float(np.mean(high_corr_ratio)),
            }
        )
    pd.DataFrame(out).to_csv(p["outputs"] / "exp2_multicollinearity_analysis.csv", index=False)


def run_figures(p: dict[str, Path], preprocessed_dir: str) -> None:
    sns.set_theme(style="whitegrid")
    fig_dir = p["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(p["outputs"] / "exp2_results_raw.csv")
    summary = pd.read_csv(p["outputs"] / "exp2_results_summary.csv")
    stability = pd.read_csv(p["outputs"] / "exp2_stability_analysis.csv") if (p["outputs"] / "exp2_stability_analysis.csv").exists() else pd.DataFrame()
    multicol = pd.read_csv(p["outputs"] / "exp2_multicollinearity_analysis.csv") if (p["outputs"] / "exp2_multicollinearity_analysis.csv").exists() else pd.DataFrame()

    # RMSE by strategy and k
    fig, ax = plt.subplots(figsize=(9, 5))
    summary_plot = summary.copy()
    summary_plot["k_label"] = (summary_plot["k_pct"] * 100).astype(int).astype(str)
    sns.barplot(data=summary_plot, x="k_label", y="mean_rmse", hue="strategy", ax=ax)
    ax.set_title("Experiment 2: RMSE by Strategy and k")
    ax.set_xlabel("k% features kept")
    ax.set_ylabel("Mean RMSE")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp2_rmse_by_strategy_k.png", dpi=300)
    plt.close(fig)

    # Execution time per strategy
    fig, ax = plt.subplots(figsize=(8, 5))
    tdf = raw.groupby("strategy", as_index=False).agg(
        mean_selection_time_s=("selection_time_s", "mean"),
        mean_total_time_s=("total_time_s", "mean"),
    )
    tlong = tdf.melt(id_vars=["strategy"], var_name="time_type", value_name="seconds")
    sns.barplot(data=tlong, x="strategy", y="seconds", hue="time_type", ax=ax)
    ax.set_title("Experiment 2: Execution Time by Strategy")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp2_time_by_strategy.png", dpi=300)
    plt.close(fig)

    # Jaccard stability
    if not stability.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        stability_plot = stability.copy()
        stability_plot["k_label"] = (stability_plot["k_pct"] * 100).astype(int).astype(str)
        sns.barplot(data=stability_plot, x="k_label", y="mean_jaccard_to_majority", hue="strategy", ax=ax)
        ax.set_title("Experiment 2: Jaccard Stability by Strategy and k")
        ax.set_xlabel("k% features kept")
        ax.set_ylabel("Mean Jaccard to Majority")
        fig.tight_layout()
        fig.savefig(fig_dir / "exp2_jaccard_by_strategy_k.png", dpi=300)
        plt.close(fig)

    # Multicollinearity behavior
    if not multicol.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        m1 = multicol.copy()
        m1["k_label"] = (m1["k_pct"] * 100).astype(int).astype(str)
        sns.barplot(data=m1, x="k_label", y="mean_abs_pairwise_corr", hue="strategy", ax=axes[0])
        axes[0].set_title("Mean Absolute Pairwise Correlation")
        axes[0].set_xlabel("k%")
        axes[0].set_ylabel("Mean |corr|")

        sns.barplot(data=m1, x="k_label", y="mean_high_corr_pair_ratio_r_ge_0_8", hue="strategy", ax=axes[1])
        axes[1].set_title("High-Correlation Pair Ratio (|r|>=0.8)")
        axes[1].set_xlabel("k%")
        axes[1].set_ylabel("Ratio")
        for i in range(2):
            if i == 1:
                axes[i].legend(title="strategy", loc="upper right")
            else:
                lg = axes[i].get_legend()
                if lg is not None:
                    lg.remove()
        fig.tight_layout()
        fig.savefig(fig_dir / "exp2_multicollinearity_by_strategy_k.png", dpi=300)
        plt.close(fig)

    # RMSE vs time trade-off
    fig, ax = plt.subplots(figsize=(8, 6))
    trade = summary.groupby("strategy", as_index=False).agg(mean_rmse=("mean_rmse", "mean"), mean_total_time_s=("mean_total_time_s", "mean"))
    sns.scatterplot(data=trade, x="mean_total_time_s", y="mean_rmse", hue="strategy", s=120, ax=ax)
    for _, r in trade.iterrows():
        ax.text(r["mean_total_time_s"], r["mean_rmse"], f" {r['strategy']}", va="center")
    ax.set_title("Experiment 2: Accuracy vs Time Trade-off")
    ax.set_xlabel("Mean Total Time (s)")
    ax.set_ylabel("Mean RMSE")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp2_rmse_vs_time_tradeoff.png", dpi=300)
    plt.close(fig)

    # SHAP plots: one per shap-based strategy x k combination
    sel = pd.read_csv(p["outputs"] / "exp2_selections_raw.csv")
    ds_all = load_all_datasets_from_local(preprocessed_dir)
    ds = [d for d in ds_all if d["name"] == "Superconductor"][0]
    X_df = ds["X"].copy()
    y = ds["y"].to_numpy()
    feature_names = list(X_df.columns)
    shap_dir = fig_dir / "exp2_shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    rep_seed = int(sel["repeat_seed"].min())
    rep_fold = int(sel["fold"].min())
    kf = KFold(n_splits=max(int(sel["fold"].max()), 2), shuffle=True, random_state=rep_seed)
    splits = list(kf.split(X_df))
    tr, te = splits[rep_fold - 1]
    X_tr_raw = X_df.iloc[tr].to_numpy()
    y_tr = y[tr]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_tr_df = pd.DataFrame(X_tr, columns=feature_names)

    shap_strategies = ["shaprfecv", "hybrid_fi_shaprfecv"]
    for strat in shap_strategies:
        for k in sorted(sel["k_pct"].unique()):
            row = sel[
                (sel["strategy"] == strat)
                & (sel["k_pct"] == k)
                & (sel["repeat_seed"] == rep_seed)
                & (sel["fold"] == rep_fold)
            ]
            if row.empty:
                continue
            selected = row.iloc[0]["selected_features"]
            cols = selected.split("|") if isinstance(selected, str) and selected else feature_names
            Xs = X_tr_df[cols]
            model = make_model(rep_seed)
            model.fit(Xs, y_tr)
            explainer = shap.TreeExplainer(model, data=Xs, feature_perturbation="interventional")
            shap_values = explainer.shap_values(Xs, check_additivity=False)

            fig = plt.figure(figsize=(8, 5))
            shap.summary_plot(shap_values, Xs, plot_type="bar", show=False)
            plt.title(f"SHAP Bar | {strat} | k={int(k*100)}%")
            plt.tight_layout()
            plt.savefig(shap_dir / f"shap_bar_{strat}_k{int(k*100)}.png", dpi=300)
            plt.close(fig)


def run_analysis_markdown(p: dict[str, Path]) -> None:
    raw = pd.read_csv(p["outputs"] / "exp2_results_raw.csv")
    summary = pd.read_csv(p["outputs"] / "exp2_results_summary.csv")
    stability = pd.read_csv(p["outputs"] / "exp2_stability_analysis.csv") if (p["outputs"] / "exp2_stability_analysis.csv").exists() else pd.DataFrame()
    multicol = pd.read_csv(p["outputs"] / "exp2_multicollinearity_analysis.csv") if (p["outputs"] / "exp2_multicollinearity_analysis.csv").exists() else pd.DataFrame()

    best = summary.loc[summary.groupby("k_pct")["mean_rmse"].idxmin(), ["k_pct", "strategy", "mean_rmse", "mean_mae", "mean_mape", "mean_selection_time_s"]]
    strat_agg = summary.groupby("strategy", as_index=False).agg(
        mean_rmse=("mean_rmse", "mean"),
        mean_mae=("mean_mae", "mean"),
        mean_mape=("mean_mape", "mean"),
        mean_time_s=("mean_total_time_s", "mean"),
    )

    lines = []
    lines.append("# Experiment 2 Results Draft")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Dataset: Superconductor")
    lines.append("- Model: XGBoost")
    lines.append("- Strategies: native_fi, shaprfecv, hybrid_fi_shaprfecv")
    lines.append(f"- Raw observations: {len(raw)}")
    lines.append("")
    lines.append("## Best Strategy by k (RMSE)")
    lines.append(best.to_markdown(index=False))
    lines.append("")
    lines.append("## Strategy Aggregates")
    lines.append(strat_agg.to_markdown(index=False))
    lines.append("")
    if not stability.empty:
        lines.append("## Stability (Jaccard)")
        lines.append(stability.sort_values(["k_pct", "mean_jaccard_to_majority"], ascending=[True, False]).to_markdown(index=False))
        lines.append("")
    if not multicol.empty:
        lines.append("## Multicollinearity Behavior")
        lines.append(multicol.sort_values(["k_pct", "mean_high_corr_pair_ratio_r_ge_0_8"]).to_markdown(index=False))
        lines.append("")
    lines.append("## Generated Figures")
    lines.append("- exp2_rmse_by_strategy_k.png")
    lines.append("- exp2_time_by_strategy.png")
    lines.append("- exp2_jaccard_by_strategy_k.png")
    lines.append("- exp2_multicollinearity_by_strategy_k.png")
    lines.append("- exp2_rmse_vs_time_tradeoff.png")
    lines.append("- exp2_shap/*.png")

    (p["outputs"] / "exp2_results_draft.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging(True)
    random.seed(SEED)
    np.random.seed(SEED)

    p = paths(Path.cwd(), run_id=args.run_id)
    p["outputs"].mkdir(parents=True, exist_ok=True)
    p["figures"].mkdir(parents=True, exist_ok=True)
    p["logs"].mkdir(parents=True, exist_ok=True)

    if args.step in ["all", "experiments"]:
        run_experiments(p, quick=args.quick, resume=args.resume, preprocessed_dir=args.use_preprocessed_dir)
        logging.info("Saved Experiment 2 raw outputs under %s", p["outputs"])
    if args.step in ["all", "summary"]:
        run_summary(p)
    if args.step in ["all", "stability"]:
        run_stability(p)
    if args.step in ["all", "multicollinearity"]:
        run_multicollinearity(p, preprocessed_dir=args.use_preprocessed_dir)
    if args.step in ["all", "figures"]:
        run_figures(p, preprocessed_dir=args.use_preprocessed_dir)
    if args.step in ["all", "analysis"]:
        run_analysis_markdown(p)


if __name__ == "__main__":
    main()
