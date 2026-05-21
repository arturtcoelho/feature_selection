from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import fcntl
import gc
import logging
import random
import time
import traceback
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.config import MODELS, SEED, STRATEGIES, build_run_config, paths
from src.feature_selection import prune_ranking_by_correlation, select_top_k_indices, strategy_rank_with_scores
from src.logging_utils import setup_logging
from src.models import make_model
from src.preprocessing import scale_train_test
from src.results_io import RAW_COLUMNS, load_raw_results, save_raw_results, save_summary, summarize_results


_DATASET_CACHE: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1 single-dataset pipeline with fast feature selection")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to CSV or ARFF dataset")
    parser.add_argument("--target-col", type=str, default=None, help="Target column name (defaults to last column)")
    parser.add_argument("--dataset-name", type=str, default="New Dataset", help="Dataset label used in outputs")
    parser.add_argument("--quick", action="store_true", help="Run quick mode: 2 folds x 2 repeats")
    parser.add_argument(
        "--step",
        choices=["all", "experiments", "summary", "stats", "stability", "figures"],
        default="all",
        help="Run one stage or all stages",
    )
    parser.add_argument("--resume", action="store_true", help="Resume experiments from existing raw CSV")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan without running")
    parser.add_argument("--workers", type=int, default=None, help="Reserved for compatibility; execution is sequential")
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
    parser.add_argument(
        "--fs-sample-size",
        type=int,
        default=2000,
        help="Max training rows used for feature selection per fold (training still uses 100%% rows)",
    )
    return parser.parse_args()


def ensure_dirs(p: dict[str, Path]) -> None:
    p["outputs"].mkdir(parents=True, exist_ok=True)
    p["figures"].mkdir(parents=True, exist_ok=True)
    p["logs"].mkdir(parents=True, exist_ok=True)


def load_single_dataset(dataset_path: str, target_col: str | None, dataset_name: str) -> dict[str, Any]:
    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise RuntimeError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".arff":
        data, _meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
    else:
        df = pd.read_csv(path)

    effective_target_col = target_col if target_col is not None else str(df.columns[-1])
    if effective_target_col not in df.columns:
        raise RuntimeError(f"Target column '{effective_target_col}' not found in {path}")

    y = pd.to_numeric(df[effective_target_col], errors="coerce")
    X = df.drop(columns=[effective_target_col]).copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
        else:
            maybe = pd.to_numeric(X[col], errors="coerce")
            maybe_series = pd.Series(maybe)
            if maybe_series.notna().mean() < 0.99:
                X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
            else:
                X[col] = maybe

    clean = X.copy()
    clean["target"] = y
    clean = clean.dropna()
    y_clean = clean.pop("target")

    logging.info("Loaded %s: rows=%d, features=%d", dataset_name, clean.shape[0], clean.shape[1])
    return {
        "name": dataset_name,
        "X": clean,
        "y": y_clean,
        "feature_names": list(clean.columns),
    }


def _key_tuple(row: dict[str, Any]) -> tuple:
    return (
        row["dataset"],
        row["model"],
        row["strategy"],
        float(row["k_pct"]),
        int(row["fold"]),
        int(row["repeat_seed"]),
    )


def _fit_predict_timed(
    model_name: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    use_gpu: bool,
) -> tuple[np.ndarray, float, float]:
    model = make_model(model_name, seed, use_gpu=use_gpu)
    try:
        t_train_0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t_train_0
    except Exception as exc:  # noqa: BLE001
        if use_gpu and model_name == "xgboost":
            logging.warning("XGBoost GPU fit failed, retrying CPU fallback: %s", exc)
            model = make_model(model_name, seed, use_gpu=False)
            t_train_0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time_s = time.perf_counter() - t_train_0
        else:
            raise
    t_pred_0 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time_s = time.perf_counter() - t_pred_0
    return preds, train_time_s, predict_time_s


def _append_csv_locked(df: pd.DataFrame, target: Path, lock_path: Path) -> None:
    if df.empty:
        return
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        file_exists = target.exists() and target.stat().st_size > 0
        with open(target, "a", encoding="utf-8", newline="") as out:
            df.to_csv(out, index=False, header=not file_exists)
            out.flush()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _init_worker(dataset: dict[str, Any]) -> None:
    global _DATASET_CACHE
    _DATASET_CACHE = dataset


def _get_cached_dataset() -> dict[str, Any]:
    if _DATASET_CACHE is None:
        raise RuntimeError("Worker dataset cache is not initialized")
    return _DATASET_CACHE


def _run_job(
    repeat_seed: int,
    fold_num: int,
    model_name: str,
    strategy: str,
    pending_k: list[float],
    folds: int,
    k_levels: list[float],
    fs_sample_size: int,
    corr_prune_threshold: float | None,
    shap_sample_ratio: float,
    use_gpu: bool,
    raw_path: str,
    sel_path: str,
    imp_path: str,
    err_path: str,
    lock_raw: str,
    lock_sel: str,
    lock_imp: str,
    lock_err: str,
) -> tuple[int, int]:
    ds = _get_cached_dataset()
    X_df = ds["X"]
    y_series = ds["y"]
    dataset_name = ds["name"]
    feature_names = np.array(ds["feature_names"])

    kf = KFold(n_splits=folds, shuffle=True, random_state=repeat_seed)
    train_idx, test_idx = list(kf.split(X_df))[fold_num - 1]

    X_train_raw = X_df.iloc[train_idx].to_numpy()
    X_test_raw = X_df.iloc[test_idx].to_numpy()
    y_train = y_series.iloc[train_idx].to_numpy()
    y_test = y_series.iloc[test_idx].to_numpy()
    X_train, X_test = scale_train_test(X_train_raw, X_test_raw)
    n_features = X_train.shape[1]

    if fs_sample_size > 0 and X_train.shape[0] > fs_sample_size:
        fs_rng = np.random.default_rng(repeat_seed * 1000 + fold_num)
        fs_idx = fs_rng.choice(X_train.shape[0], size=fs_sample_size, replace=False)
        X_train_fs = X_train[fs_idx]
        y_train_fs = y_train[fs_idx]
    else:
        X_train_fs = X_train
        y_train_fs = y_train

    baseline_metric_cache: dict[tuple, tuple[float, float, float]] = {}
    baseline_time_cache: dict[tuple, tuple[float, float, float]] = {}
    baseline_key = (dataset_name, model_name, 1.0, fold_num, repeat_seed)

    ranking = None
    ranking_pruned = None
    ranking_scores = None
    selection_time_s = 0.0
    try:
        if any(k < 1.0 for k in k_levels):
            t_sel = time.perf_counter()
            ranking, ranking_scores = strategy_rank_with_scores(
                strategy=strategy,
                X_train=X_train_fs,
                y_train=y_train_fs,
                seed=repeat_seed,
                model_name=model_name,
                shap_sample_ratio=shap_sample_ratio,
                use_gpu=use_gpu,
            )
            ranking_pruned = prune_ranking_by_correlation(ranking, X_train_fs, corr_prune_threshold)
            selection_time_s = time.perf_counter() - t_sel
            ordered_features = feature_names[ranking].tolist()
            ordered_scores = [float(ranking_scores[int(i)]) for i in ranking.tolist()]
            imp_row = {
                "dataset": dataset_name,
                "model": model_name,
                "strategy": strategy,
                "fold": fold_num,
                "repeat_seed": repeat_seed,
                "corr_prune_threshold": corr_prune_threshold,
                "ranked_features": "|".join(ordered_features),
                "ranked_scores": "|".join([f"{v:.12g}" for v in ordered_scores]),
            }
            _append_csv_locked(pd.DataFrame([imp_row]), Path(imp_path), Path(lock_imp))
    except Exception as exc:  # noqa: BLE001
        err_row = {
            "dataset": dataset_name,
            "model": model_name,
            "strategy": strategy,
            "k_pct": np.nan,
            "fold": fold_num,
            "repeat_seed": repeat_seed,
            "stage": "ranking",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _append_csv_locked(pd.DataFrame([err_row]), Path(err_path), Path(lock_err))
        return 1, 0

    wrote = 0
    for k_pct in pending_k:
        try:
            if k_pct == 1.0:
                if baseline_key not in baseline_metric_cache:
                    t_total_0 = time.perf_counter()
                    preds, train_time_s, predict_time_s = _fit_predict_timed(
                        model_name,
                        repeat_seed,
                        X_train,
                        y_train,
                        X_test,
                        use_gpu,
                    )
                    total_time_s = time.perf_counter() - t_total_0
                    mape = mean_absolute_percentage_error(y_test, preds)
                    mse = mean_squared_error(y_test, preds)
                    rmse = root_mean_squared_error(y_test, preds)
                    baseline_metric_cache[baseline_key] = (mape, mse, rmse)
                    baseline_time_cache[baseline_key] = (train_time_s, predict_time_s, total_time_s)
                mape, mse, rmse = baseline_metric_cache[baseline_key]
                train_time_s, predict_time_s, total_time_s = baseline_time_cache[baseline_key]
                selected_idx = np.arange(n_features)
                local_selection_time = 0.0
            else:
                assert ranking_pruned is not None
                selected_idx = select_top_k_indices(ranking_pruned, n_features, k_pct)
                X_train_sel = X_train[:, selected_idx]
                X_test_sel = X_test[:, selected_idx]
                t_total_0 = time.perf_counter()
                preds, train_time_s, predict_time_s = _fit_predict_timed(
                    model_name,
                    repeat_seed,
                    X_train_sel,
                    y_train,
                    X_test_sel,
                    use_gpu,
                )
                total_time_s = time.perf_counter() - t_total_0
                mape = mean_absolute_percentage_error(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                rmse = root_mean_squared_error(y_test, preds)
                local_selection_time = selection_time_s

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "strategy": strategy,
                "k_pct": float(k_pct),
                "fold": fold_num,
                "repeat_seed": repeat_seed,
                "mape": float(mape),
                "mse": float(mse),
                "rmse": float(rmse),
                "selection_time_s": float(local_selection_time),
                "train_time_s": float(train_time_s),
                "predict_time_s": float(predict_time_s),
                "total_time_s": float(total_time_s + local_selection_time),
            }
            sel = {
                "dataset": dataset_name,
                "model": model_name,
                "strategy": strategy,
                "k_pct": float(k_pct),
                "fold": fold_num,
                "repeat_seed": repeat_seed,
                "corr_prune_threshold": corr_prune_threshold,
                "n_features_total": int(n_features),
                "n_features_post_prune": int(len(ranking_pruned) if ranking_pruned is not None else n_features),
                "n_features_selected": int(len(selected_idx)),
                "selected_features": "|".join(feature_names[selected_idx].tolist()),
            }
            raw_row_df = pd.DataFrame([row]).reindex(columns=RAW_COLUMNS)
            _append_csv_locked(raw_row_df, Path(raw_path), Path(lock_raw))
            _append_csv_locked(pd.DataFrame([sel]), Path(sel_path), Path(lock_sel))
            print(
                f"done seed={repeat_seed} fold={fold_num}/{folds} model={model_name} strategy={strategy} k={k_pct}",
                flush=True,
            )
            wrote += 1
        except Exception as exc:  # noqa: BLE001
            err_row = {
                "dataset": dataset_name,
                "model": model_name,
                "strategy": strategy,
                "k_pct": float(k_pct),
                "fold": fold_num,
                "repeat_seed": repeat_seed,
                "stage": "fit_predict",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _append_csv_locked(pd.DataFrame([err_row]), Path(err_path), Path(lock_err))
        finally:
            gc.collect()

    return 1, wrote


def run_experiments_single_dataset(
    dataset: dict[str, Any],
    run_config,
    existing_raw: pd.DataFrame,
    fs_sample_size: int,
    p: dict[str, Path],
) -> None:
    done = set()
    if not existing_raw.empty:
        for _, row in existing_raw.iterrows():
            done.add(_key_tuple(row.to_dict()))

    jobs: list[tuple[int, int, str, str, list[float]]] = []
    for repeat_seed in run_config.repeat_seeds:
        for fold_num in range(1, run_config.folds + 1):
            for model_name in MODELS:
                for strategy in STRATEGIES:
                    pending_k = [
                        k
                        for k in run_config.k_levels
                        if (dataset["name"], model_name, strategy, float(k), int(fold_num), int(repeat_seed)) not in done
                    ]
                    if pending_k:
                        jobs.append((repeat_seed, fold_num, model_name, strategy, pending_k))

    logging.info("Planned jobs=%d, workers=%d", len(jobs), run_config.workers)
    if not jobs:
        logging.info("Nothing to run. Resume found all rows already computed.")
        return

    err_path = p["outputs"] / "errors_raw.csv"
    lock_raw = p["outputs"] / "results_raw.csv.lock"
    lock_sel = p["outputs"] / "selections_raw.csv.lock"
    lock_imp = p["outputs"] / "feature_importances_raw.csv.lock"
    lock_err = p["outputs"] / "errors_raw.csv.lock"

    pbar = tqdm(total=len(jobs), desc="Jobs", unit="job")
    if run_config.workers <= 1:
        _init_worker(dataset)
        for repeat_seed, fold_num, model_name, strategy, pending_k in jobs:
            done_job, _wrote = _run_job(
                repeat_seed,
                fold_num,
                model_name,
                strategy,
                pending_k,
                run_config.folds,
                run_config.k_levels,
                fs_sample_size,
                run_config.corr_prune_threshold,
                run_config.shap_sample_ratio,
                run_config.use_gpu,
                str(p["raw"]),
                str(p["selections"]),
                str(p["importances"]),
                str(err_path),
                str(lock_raw),
                str(lock_sel),
                str(lock_imp),
                str(lock_err),
            )
            pbar.update(done_job)
    else:
        with ProcessPoolExecutor(max_workers=run_config.workers, initializer=_init_worker, initargs=(dataset,)) as ex:
            futures = [
                ex.submit(
                    _run_job,
                    repeat_seed,
                    fold_num,
                    model_name,
                    strategy,
                    pending_k,
                    run_config.folds,
                    run_config.k_levels,
                    fs_sample_size,
                    run_config.corr_prune_threshold,
                    run_config.shap_sample_ratio,
                    run_config.use_gpu,
                    str(p["raw"]),
                    str(p["selections"]),
                    str(p["importances"]),
                    str(err_path),
                    str(lock_raw),
                    str(lock_sel),
                    str(lock_imp),
                    str(lock_err),
                )
                for repeat_seed, fold_num, model_name, strategy, pending_k in jobs
            ]
            for fut in as_completed(futures):
                done_job, _wrote = fut.result()
                pbar.update(done_job)
    pbar.close()


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
        "Run config: quick=%s, folds=%d, repeats=%s, k=%s, corr_prune_threshold=%s, fs_sample_size=%d, run_root=%s",
        cfg.quick,
        cfg.folds,
        cfg.repeat_seeds,
        cfg.k_levels,
        cfg.corr_prune_threshold,
        args.fs_sample_size,
        p["root"],
    )

    if args.dry_run:
        est = 1 * 3 * 4 * len(cfg.k_levels) * cfg.folds * len(cfg.repeat_seeds)
        print(f"Dry run plan: estimated observations={est}")
        return

    raw_df = load_raw_results(p["raw"]) if args.resume else load_raw_results(Path("__nonexistent__.csv"))

    if args.step in ["all", "experiments"]:
        t_exp_0 = time.perf_counter()
        dataset = load_single_dataset(args.dataset_path, args.target_col, args.dataset_name)
        run_experiments_single_dataset(
            dataset=dataset,
            run_config=cfg,
            existing_raw=raw_df,
            fs_sample_size=max(1, int(args.fs_sample_size)),
            p=p,
        )
        raw_df = load_raw_results(p["raw"])
        save_raw_results(p["raw"], raw_df)
        logging.info("Saved raw results to %s", p["raw"])
        logging.info("Total experiment stage time: %.3f s", time.perf_counter() - t_exp_0)

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
