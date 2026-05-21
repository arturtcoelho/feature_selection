from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

from src.config import SEED, build_run_config, paths
from src.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1.1: Exp1 protocol on Allstate only")
    parser.add_argument("--quick", action="store_true", help="Run quick mode")
    parser.add_argument(
        "--step",
        choices=["all", "experiments", "summary", "stats", "stability", "figures"],
        default="all",
        help="Run one stage or all stages",
    )
    parser.add_argument("--resume", action="store_true", help="Resume experiments from existing raw CSV")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan without running")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process workers for model-level parallelism (default 1 for visible progress logs)",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode for GPU-capable models")
    parser.add_argument(
        "--metric",
        choices=["mape", "rmse", "mse"],
        default="rmse",
        help="Metric used in statistical tests and figure axes",
    )
    parser.add_argument(
        "--corr-prune-threshold",
        type=float,
        default=None,
        help="Optional correlation-pruning threshold in (0,1), applied after ranking and before top-k.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Isolate outputs under runs/<run-id>/ (safe, non-overwriting workflow).",
    )
    parser.add_argument("--arff-path", type=str, default="dataset.arff", help="Path to Allstate ARFF file")
    parser.add_argument("--target-col", type=str, default=None, help="Target column name; default=last ARFF column")
    return parser.parse_args()


def ensure_dirs(p: dict[str, Path]) -> None:
    p["outputs"].mkdir(parents=True, exist_ok=True)
    p["figures"].mkdir(parents=True, exist_ok=True)
    p["logs"].mkdir(parents=True, exist_ok=True)


def load_allstate_dataset(arff_path: str, target_col: str | None) -> list[dict]:
    path = Path(arff_path)
    if not path.exists():
        raise RuntimeError(f"Missing ARFF dataset file: {path}")

    data, _meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)

    tgt = target_col if target_col is not None else df.columns[-1]
    if tgt not in df.columns:
        raise RuntimeError(f"Target column '{tgt}' missing in {path}")

    y = pd.to_numeric(df[tgt], errors="coerce")
    X = df.drop(columns=[tgt]).copy()

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
        else:
            maybe_num = pd.to_numeric(X[col], errors="coerce")
            if maybe_num.notna().mean() < 0.99:
                X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
            else:
                X[col] = maybe_num

    clean = X.copy()
    clean["target"] = y
    clean = clean.dropna()
    y_clean = clean.pop("target")
    X_clean = clean

    logging.info("Loaded Allstate: rows=%d, features=%d", X_clean.shape[0], X_clean.shape[1])
    return [
        {
            "name": "Allstate",
            "X": X_clean,
            "y": y_clean,
            "feature_names": list(X_clean.columns),
        }
    ]


def main() -> None:
    args = parse_args()
    setup_logging(verbose=True)

    random.seed(SEED)
    np.random.seed(SEED)

    p = paths(Path.cwd(), run_id=args.run_id)
    ensure_dirs(p)

    if args.run_id and p["raw"].exists() and not args.resume:
        raise RuntimeError(
            f"Run '{args.run_id}' already has results at {p['raw']}. "
            "Use --resume to continue it, or choose a new --run-id."
        )
    cfg = build_run_config(quick=args.quick)
    cfg = replace(cfg, workers=max(1, args.workers))
    if args.no_gpu:
        cfg = replace(cfg, use_gpu=False)
    if args.corr_prune_threshold is not None:
        cfg = replace(cfg, corr_prune_threshold=args.corr_prune_threshold)

    logging.info(
        "Run config: quick=%s, folds=%d, repeats=%s, k=%s, corr_prune_threshold=%s, run_root=%s",
        cfg.quick,
        cfg.folds,
        cfg.repeat_seeds,
        cfg.k_levels,
        cfg.corr_prune_threshold,
        p["root"],
    )
    logging.info("Dataset scope locked: Allstate only")
    logging.info("Model grid: ridge + extratrees + xgboost")
    logging.info("Feature-selection grid: mi + rfe + tree + shap")
    logging.info("GPU note: only xgboost uses GPU; ridge/extratrees are CPU models")
    est = 1 * 3 * 4 * len(cfg.k_levels) * cfg.folds * len(cfg.repeat_seeds)
    logging.info("Planned observations: %d", est)
    if args.dry_run:
        print(f"Dry run plan: estimated observations={est}")
        return

    from src.results_io import load_raw_results, save_raw_results, save_summary, summarize_results

    raw_df = load_raw_results(p["raw"]) if args.resume else load_raw_results(Path("__nonexistent__.csv"))
    if args.step in ["all", "experiments"]:
        from src.experiment_runner import run_experiments

        t_exp_0 = time.perf_counter()
        datasets = load_allstate_dataset(args.arff_path, args.target_col)

        def _checkpoint(rows, sels, imps):
            if rows:
                partial_new = pd.DataFrame(rows)
                merged = pd.concat([raw_df, partial_new], ignore_index=True)
                merged.to_csv(p["raw"], index=False)
            if sels:
                pd.DataFrame(sels).to_csv(p["selections"], index=False)
            if imps:
                pd.DataFrame(imps).to_csv(p["importances"], index=False)

        raw_df, selection_df, importance_df = run_experiments(
            datasets,
            cfg,
            raw_df,
            checkpoint_every_tasks=1,
            checkpoint_callback=_checkpoint,
        )
        exp_total_s = time.perf_counter() - t_exp_0
        save_raw_results(p["raw"], raw_df)
        if not selection_df.empty:
            selection_df.to_csv(p["selections"], index=False)
        if not importance_df.empty:
            importance_df.to_csv(p["importances"], index=False)
        logging.info("Saved raw results to %s", p["raw"])
        logging.info("Total experiment stage time: %.3f s", exp_total_s)

    if args.step in ["all", "summary"]:
        raw_df = load_raw_results(p["raw"])
        summary_df = summarize_results(raw_df)
        save_summary(p["summary"], summary_df)
        logging.info("Saved summary results to %s", p["summary"])

    if args.step in ["all", "stats"]:
        from src.stats_tests import run_statistical_tests

        raw_df = load_raw_results(p["raw"])
        stats_df = run_statistical_tests(raw_df, alpha=cfg.alpha, metric_col=args.metric)
        stats_df.to_csv(p["stats"], index=False)
        logging.info("Saved stats tests to %s", p["stats"])

    if args.step in ["all", "stability"]:
        from src.stability import compute_stability_analysis

        if not p["selections"].exists():
            logging.warning("No selections file found at %s, skipping stability", p["selections"])
        else:
            sel_df = pd.read_csv(p["selections"])
            stability_df = compute_stability_analysis(sel_df)
            stability_df.to_csv(p["stability"], index=False)
            logging.info("Saved stability analysis to %s", p["stability"])

    if args.step in ["all", "figures"]:
        from src.figures import generate_figures
        from src.stability import compute_shap_tree_overlap

        raw_df = load_raw_results(p["raw"])
        if p["selections"].exists():
            sel_df = pd.read_csv(p["selections"])
            overlap_df = compute_shap_tree_overlap(sel_df, k_pct=0.5)
        else:
            overlap_df = raw_df.iloc[0:0].copy()
            overlap_df["dataset"] = []
            overlap_df["jaccard"] = []
        generate_figures(raw_df, overlap_df, p["figures"], metric_col=args.metric)
        logging.info("Saved figures to %s", p["figures"])


if __name__ == "__main__":
    main()
