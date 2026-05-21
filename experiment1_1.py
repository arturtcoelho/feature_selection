from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import fcntl
import logging
from pathlib import Path
import random
import time
import traceback

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.config import SEED, STRATEGIES, MODELS, build_run_config, paths
from src.feature_selection import prune_ranking_by_correlation, select_top_k_indices, strategy_rank_with_scores
from src.logging_utils import setup_logging
from src.models import make_model
from src.preprocessing import scale_train_test
from src.results_io import RAW_COLUMNS, load_raw_results, save_raw_results, save_summary, summarize_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 1.1: Exp1 protocol on Allstate only")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--step", choices=["all", "experiments", "summary", "stats", "stability", "figures"], default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--metric", choices=["mape", "rmse", "mse"], default="rmse")
    p.add_argument("--corr-prune-threshold", type=float, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--arff-path", type=str, default="dataset.arff")
    p.add_argument("--target-col", type=str, default=None)
    return p.parse_args()


def ensure_dirs(pmap: dict[str, Path]) -> None:
    pmap["outputs"].mkdir(parents=True, exist_ok=True)
    pmap["figures"].mkdir(parents=True, exist_ok=True)
    pmap["logs"].mkdir(parents=True, exist_ok=True)


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


def _load_allstate(arff_path: str, target_col: str | None) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    path = Path(arff_path)
    if not path.exists():
        raise RuntimeError(f"Missing ARFF dataset file: {path}")
    data, _meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
    tgt = target_col if target_col is not None else str(df.columns[-1])
    y = pd.to_numeric(df[tgt], errors="coerce")
    X = df.drop(columns=[tgt]).copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
        else:
            maybe = pd.to_numeric(X[col], errors="coerce")
            if maybe.notna().mean() < 0.99:
                X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
            else:
                X[col] = maybe
    clean = X.copy()
    clean["target"] = y
    clean = clean.dropna()
    y_clean = clean.pop("target")
    return clean, y_clean, list(clean.columns)


def _fit_predict(model_name: str, seed: int, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, use_gpu: bool):
    model = make_model(model_name, seed, use_gpu=use_gpu)
    try:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_s = time.perf_counter() - t0
    except Exception:
        if use_gpu and model_name == "xgboost":
            model = make_model(model_name, seed, use_gpu=False)
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_s = time.perf_counter() - t0
        else:
            raise
    t1 = time.perf_counter()
    pred = model.predict(X_test)
    pred_s = time.perf_counter() - t1
    return pred, train_s, pred_s


def _run_job(
    seed: int,
    fold: int,
    model_name: str,
    strategy: str,
    folds: int,
    k_levels: list[float],
    done: set[tuple],
    arff_path: str,
    target_col: str | None,
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
    X_df, y_s, feature_names = _load_allstate(arff_path, target_col)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    tr, te = list(kf.split(X_df))[fold - 1]
    X_tr_raw = X_df.iloc[tr].to_numpy()
    X_te_raw = X_df.iloc[te].to_numpy()
    y_tr = y_s.iloc[tr].to_numpy()
    y_te = y_s.iloc[te].to_numpy()
    X_tr, X_te = scale_train_test(X_tr_raw, X_te_raw)
    n_features = X_tr.shape[1]

    rows = []
    sels = []
    imps = []
    errs = []

    print(f"job seed={seed} fold={fold}/{folds} model={model_name} strategy={strategy}", flush=True)

    ranking = None
    ranking_pruned = None
    ranking_scores = None
    selection_time_s = 0.0
    try:
        t_sel = time.perf_counter()
        ranking, ranking_scores = strategy_rank_with_scores(
            strategy=strategy,
            X_train=X_tr,
            y_train=y_tr,
            seed=seed,
            model_name=model_name,
            shap_sample_ratio=shap_sample_ratio,
            use_gpu=use_gpu,
        )
        ranking_pruned = prune_ranking_by_correlation(ranking, X_tr, corr_prune_threshold)
        selection_time_s = time.perf_counter() - t_sel
        ordered_features = np.array(feature_names)[ranking].tolist()
        ordered_scores = [float(ranking_scores[int(i)]) for i in ranking.tolist()]
        imps.append(
            {
                "dataset": "Allstate",
                "model": model_name,
                "strategy": strategy,
                "fold": fold,
                "repeat_seed": seed,
                "corr_prune_threshold": corr_prune_threshold,
                "ranked_features": "|".join(ordered_features),
                "ranked_scores": "|".join([f"{v:.12g}" for v in ordered_scores]),
            }
        )
        _append_csv_locked(pd.DataFrame(imps[-1:]), Path(imp_path), Path(lock_imp))
    except Exception as exc:  # noqa: BLE001
        errs.append(
            {
                "dataset": "Allstate",
                "model": model_name,
                "strategy": strategy,
                "k_pct": np.nan,
                "fold": fold,
                "repeat_seed": seed,
                "stage": "ranking",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _append_csv_locked(pd.DataFrame(errs[-1:]), Path(err_path), Path(lock_err))
        return 1, 0

    wrote = 0
    for i, k_pct in enumerate(k_levels, start=1):
        print(f"k {i}/{len(k_levels)} ({int(k_pct*100)}%)", flush=True)
        key = ("Allstate", model_name, strategy, float(k_pct), int(fold), int(seed))
        if key in done:
            continue
        try:
            if k_pct == 1.0:
                idx = np.arange(n_features)
                local_sel_s = 0.0
                Xtr_sel, Xte_sel = X_tr, X_te
            else:
                assert ranking_pruned is not None
                idx = select_top_k_indices(ranking_pruned, n_features, k_pct)
                local_sel_s = selection_time_s
                Xtr_sel, Xte_sel = X_tr[:, idx], X_te[:, idx]

            t0 = time.perf_counter()
            pred, train_s, pred_s = _fit_predict(model_name, seed, Xtr_sel, y_tr, Xte_sel, use_gpu)
            total_s = (time.perf_counter() - t0) + local_sel_s
            mape = float(mean_absolute_percentage_error(y_te, pred))
            mse = float(mean_squared_error(y_te, pred))
            rmse = float(root_mean_squared_error(y_te, pred))
            row = {
                "dataset": "Allstate",
                "model": model_name,
                "strategy": strategy,
                "k_pct": float(k_pct),
                "fold": fold,
                "repeat_seed": seed,
                "mape": mape,
                "mse": mse,
                "rmse": rmse,
                "selection_time_s": float(local_sel_s),
                "train_time_s": float(train_s),
                "predict_time_s": float(pred_s),
                "total_time_s": float(total_s),
            }
            sel = {
                "dataset": "Allstate",
                "model": model_name,
                "strategy": strategy,
                "k_pct": float(k_pct),
                "fold": fold,
                "repeat_seed": seed,
                "corr_prune_threshold": corr_prune_threshold,
                "n_features_total": int(n_features),
                "n_features_post_prune": int(len(ranking_pruned) if ranking_pruned is not None else n_features),
                "n_features_selected": int(len(idx)),
                "selected_features": "|".join(np.array(feature_names)[idx].tolist()),
            }
            rows.append(row)
            sels.append(sel)
            _append_csv_locked(pd.DataFrame([row])[RAW_COLUMNS], Path(raw_path), Path(lock_raw))
            _append_csv_locked(pd.DataFrame([sel]), Path(sel_path), Path(lock_sel))
            wrote += 1
        except Exception as exc:  # noqa: BLE001
            errs.append(
                {
                    "dataset": "Allstate",
                    "model": model_name,
                    "strategy": strategy,
                    "k_pct": float(k_pct),
                    "fold": fold,
                    "repeat_seed": seed,
                    "stage": "fit_predict",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            _append_csv_locked(pd.DataFrame(errs[-1:]), Path(err_path), Path(lock_err))
    return 1, wrote


def run_experiments_allstate(pmap: dict[str, Path], cfg, args: argparse.Namespace) -> None:
    repeats = cfg.repeat_seeds
    folds = cfg.folds
    k_levels = cfg.k_levels
    raw_path = pmap["raw"]
    sel_path = pmap["selections"]
    imp_path = pmap["importances"]
    err_path = pmap["outputs"] / "errors_raw.csv"
    lock_raw = pmap["outputs"] / "results_raw.csv.lock"
    lock_sel = pmap["outputs"] / "selections_raw.csv.lock"
    lock_imp = pmap["outputs"] / "feature_importances_raw.csv.lock"
    lock_err = pmap["outputs"] / "errors_raw.csv.lock"

    existing = pd.read_csv(raw_path) if (args.resume and raw_path.exists()) else pd.DataFrame(columns=RAW_COLUMNS)
    done = set()
    if not existing.empty:
        for _, r in existing.iterrows():
            done.add((str(r["dataset"]), str(r["model"]), str(r["strategy"]), float(r["k_pct"]), int(r["fold"]), int(r["repeat_seed"])))

    jobs = [(seed, fold, model, strategy) for seed in repeats for fold in range(1, folds + 1) for model in MODELS for strategy in STRATEGIES]
    total = len(jobs)
    logging.info("Planned jobs=%d workers=%d", total, cfg.workers)

    pbar = tqdm(total=total, desc="Exp1.1 jobs", unit="job")
    if cfg.workers <= 1:
        for seed, fold, model, strategy in jobs:
            done_job, wrote = _run_job(
                seed, fold, model, strategy, folds, k_levels, done,
                args.arff_path, args.target_col, cfg.corr_prune_threshold, cfg.shap_sample_ratio, cfg.use_gpu,
                str(raw_path), str(sel_path), str(imp_path), str(err_path),
                str(lock_raw), str(lock_sel), str(lock_imp), str(lock_err),
            )
            pbar.update(done_job)
            logging.info("job done seed=%d fold=%d model=%s strategy=%s rows=%d", seed, fold, model, strategy, wrote)
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futs = [
                ex.submit(
                    _run_job,
                    seed, fold, model, strategy, folds, k_levels, done,
                    args.arff_path, args.target_col, cfg.corr_prune_threshold, cfg.shap_sample_ratio, cfg.use_gpu,
                    str(raw_path), str(sel_path), str(imp_path), str(err_path),
                    str(lock_raw), str(lock_sel), str(lock_imp), str(lock_err),
                )
                for seed, fold, model, strategy in jobs
            ]
            for fut in as_completed(futs):
                done_job, _wrote = fut.result()
                pbar.update(done_job)
    pbar.close()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=True)
    random.seed(SEED)
    np.random.seed(SEED)

    pmap = paths(Path.cwd(), run_id=args.run_id)
    ensure_dirs(pmap)
    cfg = build_run_config(quick=args.quick)
    cfg = replace(cfg, workers=max(1, int(args.workers)))
    if args.no_gpu:
        cfg = replace(cfg, use_gpu=False)
    if args.corr_prune_threshold is not None:
        cfg = replace(cfg, corr_prune_threshold=args.corr_prune_threshold)

    est = 1 * len(MODELS) * len(STRATEGIES) * len(cfg.k_levels) * cfg.folds * len(cfg.repeat_seeds)
    logging.info("Run root=%s", pmap["root"])
    logging.info("Planned observations=%d", est)
    if args.dry_run:
        print(f"Dry run plan: estimated observations={est}")
        return

    if args.step in ["all", "experiments"]:
        run_experiments_allstate(pmap, cfg, args)

    if args.step in ["all", "summary"]:
        raw_df = load_raw_results(pmap["raw"])
        save_summary(pmap["summary"], summarize_results(raw_df))

    if args.step in ["all", "stats"]:
        from src.stats_tests import run_statistical_tests

        raw_df = load_raw_results(pmap["raw"])
        run_statistical_tests(raw_df, alpha=cfg.alpha, metric_col=args.metric).to_csv(pmap["stats"], index=False)

    if args.step in ["all", "stability"]:
        from src.stability import compute_stability_analysis

        if pmap["selections"].exists():
            pd.read_csv(pmap["selections"]).pipe(compute_stability_analysis).to_csv(pmap["stability"], index=False)

    if args.step in ["all", "figures"]:
        from src.figures import generate_figures
        from src.stability import compute_shap_tree_overlap

        raw_df = load_raw_results(pmap["raw"])
        if pmap["selections"].exists():
            overlap_df = compute_shap_tree_overlap(pd.read_csv(pmap["selections"]), k_pct=0.5)
        else:
            overlap_df = pd.DataFrame(columns=["dataset", "jaccard"])
        generate_figures(raw_df, overlap_df, pmap["figures"], metric_col=args.metric)
        save_raw_results(pmap["raw"], raw_df)


if __name__ == "__main__":
    main()
