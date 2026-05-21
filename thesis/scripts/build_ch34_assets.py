import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parent / "runs"
OUT = ROOT / "assets" / "generated"
OUT.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_exp1_summary() -> dict:
    df = _read_csv(RUNS / "run01_corr09" / "outputs" / "results_raw.csv")
    agg = (
        df.groupby(["dataset", "model", "strategy"], as_index=False)[
            ["rmse", "mape", "selection_time_s", "total_time_s"]
        ]
        .mean()
        .sort_values(["dataset", "model", "rmse"])
    )
    winners = agg.loc[agg.groupby(["dataset", "model"])["rmse"].idxmin()].copy()
    winners = winners.sort_values(["dataset", "model"])
    ranks = agg.copy()
    ranks["rank_rmse"] = ranks.groupby(["dataset", "model"])["rmse"].rank(method="average")
    rank_mean = ranks.groupby("strategy", as_index=False)["rank_rmse"].mean().sort_values("rank_rmse")

    agg.to_csv(OUT / "exp1_dataset_model_strategy_summary.csv", index=False)
    winners.to_csv(OUT / "exp1_winners_by_dataset_model.csv", index=False)
    rank_mean.to_csv(OUT / "exp1_strategy_mean_rank.csv", index=False)

    return {
        "winner_counts": winners["strategy"].value_counts().to_dict(),
        "rank_mean": {r.strategy: float(r.rank_rmse) for r in rank_mean.itertuples(index=False)},
    }


def _dataset_alias(name: str) -> str:
    return "Exp2A-Superconductor" if name == "Superconductor" else "Exp2B-Allstate"


def build_unified_exp2() -> dict:
    exp2 = _read_csv(RUNS / "exp2_run12" / "outputs" / "exp2_results_raw.csv")
    exp3 = _read_csv(RUNS / "exp3_run01" / "outputs" / "exp3_results_raw.csv")
    sel2 = _read_csv(RUNS / "exp2_run12" / "outputs" / "exp2_selections_raw.csv")
    sel3 = _read_csv(RUNS / "exp3_run01" / "outputs" / "exp3_selections_raw.csv")

    merged = pd.concat([exp2, exp3], ignore_index=True)
    merged["dataset_instance"] = merged["dataset"].map(_dataset_alias)

    summary = (
        merged.groupby(["dataset_instance", "k_pct", "strategy"], as_index=False)[
            ["rmse", "mae", "mape", "selection_time_s", "train_time_s", "predict_time_s", "total_time_s"]
        ]
        .mean()
        .sort_values(["dataset_instance", "k_pct", "strategy"])
    )
    summary.to_csv(OUT / "exp2_unified_summary_by_k_strategy.csv", index=False)

    # compute overhead ratios against native_fi at each k for each dataset instance
    ratio_rows = []
    for (ds, k), g in summary.groupby(["dataset_instance", "k_pct"]):
        g = g.set_index("strategy")
        if "native_fi" not in g.index:
            continue
        base_sel = g.loc["native_fi", "selection_time_s"]
        base_total = g.loc["native_fi", "total_time_s"]
        for strategy, row in g.iterrows():
            ratio_rows.append(
                {
                    "dataset_instance": ds,
                    "k_pct": k,
                    "strategy": strategy,
                    "selection_time_ratio_vs_native_fi": (row["selection_time_s"] / base_sel) if base_sel > 0 else None,
                    "total_time_ratio_vs_native_fi": row["total_time_s"] / base_total if base_total > 0 else None,
                }
            )
    pd.DataFrame(ratio_rows).to_csv(OUT / "exp2_unified_time_ratios.csv", index=False)

    # Deep-dive candidates: near-equal rmse with low feature overlap
    keys = ["dataset", "model", "strategy", "k_pct", "fold", "repeat_seed"]
    sel = pd.concat([sel2, sel3], ignore_index=True)
    m = merged.merge(sel[keys + ["selected_features"]], on=keys, how="inner")

    def parse_set(v: str):
        if pd.isna(v) or str(v).strip() == "":
            return set()
        return set(str(v).split("|"))

    rows = []
    strategies = ["native_fi", "custom_shap_rfe", "hybrid_fi_custom_shap_rfe"]
    for (dataset, k, fold, seed), g in m.groupby(["dataset", "k_pct", "fold", "repeat_seed"]):
        rec = {r.strategy: r for r in g.itertuples(index=False)}
        pairs = [("native_fi", "custom_shap_rfe"), ("native_fi", "hybrid_fi_custom_shap_rfe"), ("custom_shap_rfe", "hybrid_fi_custom_shap_rfe")]
        for a, b in pairs:
            if a not in rec or b not in rec:
                continue
            ra, rb = rec[a], rec[b]
            sa, sb = parse_set(ra.selected_features), parse_set(rb.selected_features)
            union = sa | sb
            inter = sa & sb
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_instance": _dataset_alias(dataset),
                    "k_pct": k,
                    "fold": fold,
                    "repeat_seed": seed,
                    "strategy_a": a,
                    "strategy_b": b,
                    "rmse_a": ra.rmse,
                    "rmse_b": rb.rmse,
                    "rmse_abs_diff": abs(ra.rmse - rb.rmse),
                    "rmse_rel_diff": abs(ra.rmse - rb.rmse) / max(abs(ra.rmse), 1e-12),
                    "mae_abs_diff": abs(ra.mae - rb.mae),
                    "jaccard": len(inter) / len(union) if union else 1.0,
                    "n_sym_diff": len(sa ^ sb),
                    "n_features_a": len(sa),
                    "n_features_b": len(sb),
                }
            )
    cases = pd.DataFrame(rows)
    cases = cases.sort_values(["rmse_rel_diff", "jaccard", "n_sym_diff"], ascending=[True, True, False])
    cases.to_csv(OUT / "exp2_unified_case_candidates.csv", index=False)

    # pick 3 showcase cases, forcing both datasets and k around 0.10-0.15 where possible
    showcase = []
    wanted = [
        ("Superconductor", 0.10, "native_fi", "custom_shap_rfe"),
        ("Allstate", 0.10, "native_fi", "custom_shap_rfe"),
        ("Superconductor", 0.15, "native_fi", "custom_shap_rfe"),
    ]
    for ds, k, a, b in wanted:
        sub = cases[
            (cases["dataset"] == ds)
            & (cases["k_pct"] == k)
            & (cases["strategy_a"] == a)
            & (cases["strategy_b"] == b)
        ]
        if not sub.empty:
            showcase.append(sub.iloc[0])
    if len(showcase) < 3:
        extra = cases.head(3 - len(showcase))
        showcase.extend([r for _, r in extra.iterrows()])
    showcase_df = pd.DataFrame(showcase).drop_duplicates()
    showcase_df.to_csv(OUT / "exp2_unified_showcase_cases.csv", index=False)

    return {
        "n_rows_unified": int(len(merged)),
        "n_showcase": int(len(showcase_df)),
    }


def build_figures() -> None:
    summary = _read_csv(OUT / "exp2_unified_summary_by_k_strategy.csv")

    for metric, ylab, fname in [
        ("rmse", "RMSE", "fig_ch4_rmse_vs_k_unified.png"),
        ("total_time_s", "Total time (s)", "fig_ch4_total_time_vs_k_unified.png"),
        ("selection_time_s", "Selection time (s)", "fig_ch4_selection_time_vs_k_unified.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        for ax, ds in zip(axes, ["Exp2A-Superconductor", "Exp2B-Allstate"]):
            sub = summary[summary["dataset_instance"] == ds]
            for strategy in sorted(sub["strategy"].unique()):
                s = sub[sub["strategy"] == strategy].sort_values("k_pct")
                ax.plot(s["k_pct"], s[metric], marker="o", label=strategy)
            ax.set_title(ds)
            ax.set_xlabel("k (fraction of retained features)")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
        axes[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=160)
        plt.close(fig)

    # case-study scatter: rmse relative diff vs overlap
    cases = _read_csv(OUT / "exp2_unified_case_candidates.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for ds, marker in [("Exp2A-Superconductor", "o"), ("Exp2B-Allstate", "s")]:
        sub = cases[cases["dataset_instance"] == ds]
        ax.scatter(sub["jaccard"], sub["rmse_rel_diff"], alpha=0.6, marker=marker, label=ds)
    ax.set_xlabel("Feature-set overlap (Jaccard)")
    ax.set_ylabel("Relative RMSE difference")
    ax.set_title("Near-equivalent accuracy with different selected subsets")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_ch4_case_scatter_overlap_vs_rmse.png", dpi=160)
    plt.close(fig)


def main() -> None:
    exp1 = build_exp1_summary()
    exp2u = build_unified_exp2()
    build_figures()
    payload = {"exp1": exp1, "exp2_unified": exp2u}
    (OUT / "analysis_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
