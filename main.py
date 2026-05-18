from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from src.config import SEED, build_run_config, paths
from src.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masters thesis experiment pipeline")
    parser.add_argument("--quick", action="store_true", help="Run quick mode: 2 folds x 2 repeats x k=[50,100]")
    parser.add_argument(
        "--step",
        choices=["all", "experiments", "summary", "stats", "stability", "figures"],
        default="all",
        help="Run one stage or all stages",
    )
    parser.add_argument("--resume", action="store_true", help="Resume experiments from existing raw CSV")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan without running")
    parser.add_argument("--workers", type=int, default=None, help="Process workers for model-level parallelism")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode for GPU-capable models")
    parser.add_argument(
        "--use-preprocessed-dir",
        type=str,
        default=None,
        help="Use local preprocessed CSV directory instead of downloading",
    )
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
    return parser.parse_args()


def ensure_dirs(p: dict[str, Path]) -> None:
    p["outputs"].mkdir(parents=True, exist_ok=True)
    p["figures"].mkdir(parents=True, exist_ok=True)
    p["logs"].mkdir(parents=True, exist_ok=True)


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
    if args.workers is not None:
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
    if args.dry_run:
        est = 3 * 3 * 4 * len(cfg.k_levels) * cfg.folds * len(cfg.repeat_seeds)
        print(f"Dry run plan: estimated observations={est}")
        return

    from src.results_io import load_raw_results, save_raw_results, save_summary, summarize_results

    raw_df = load_raw_results(p["raw"]) if args.resume else load_raw_results(Path("__nonexistent__.csv"))
    if args.step in ["all", "experiments"]:
        from src.data_loading import load_all_datasets, load_all_datasets_from_local
        from src.experiment_runner import run_experiments

        t_exp_0 = time.perf_counter()
        if args.use_preprocessed_dir:
            datasets = load_all_datasets_from_local(args.use_preprocessed_dir)
        else:
            datasets = load_all_datasets()

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

    if p["summary"].exists():
        from pandas import read_csv
        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None

        summary_df = read_csv(p["summary"])
        if tabulate is not None:
            print(tabulate(summary_df.head(20), headers="keys", tablefmt="github", showindex=False))
        else:
            print(summary_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
