from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import logging
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import DMatrix, XGBRegressor
from scipy.io import arff

from src.config import SEED, paths
from src.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 3: Allstate ARFF + XGBoost + custom SHAP-RFE")
    p.add_argument("--step", choices=["all", "experiments", "summary", "stability", "multicollinearity", "figures", "analysis"], default="all")
    p.add_argument("--run-id", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--arff-path", default="dataset.arff")
    p.add_argument("--target-col", default=None, help="Target column name; defaults to last column")
    p.add_argument("--shap-step", type=int, default=5)
    p.add_argument("--shap-sample-ratio", type=float, default=0.10)
    p.add_argument("--shap-max-samples", type=int, default=1600)
    p.add_argument("--xgb-jobs", type=int, default=2, help="Threads per XGBoost fit (set low when using workers)")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU mode for XGBoost")
    return p.parse_args()


def make_model(seed: int, xgb_jobs: int, use_gpu: bool) -> XGBRegressor:
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "random_state": seed,
        "verbosity": 0,
        "n_jobs": max(1, int(xgb_jobs)),
        "tree_method": "hist",
        "enable_categorical": True,
    }
    if use_gpu:
        params["device"] = "cuda"
    return XGBRegressor(
        **params,
    )


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = mean_absolute_percentage_error(y_true, y_pred)
    return float(m) if np.isfinite(m) else np.nan


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _append_csv_locked(df: pd.DataFrame, target: Path, lock_path: Path) -> None:
    if df.empty:
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        file_exists = target.exists() and target.stat().st_size > 0
        with open(target, "a", encoding="utf-8", newline="") as out:
            df.to_csv(out, index=False, header=not file_exists)
            out.flush()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_allstate_arff(arff_path: str, target_col: str | None) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
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
            X[col] = X[col].astype("category")
        else:
            maybe_num = pd.to_numeric(X[col], errors="coerce")
            if maybe_num.notna().mean() < 0.99:
                X[col] = X[col].astype("category")
            else:
                X[col] = maybe_num

    clean = X.copy()
    clean["target"] = y
    clean = clean.dropna()
    y_clean = clean.pop("target").to_numpy()
    X_clean = clean
    return X_clean, y_clean, list(X_clean.columns)


def _is_valid_k_for_strategy(strategy: str, k: float) -> bool:
    if strategy == "native_fi":
        return True
    return k < 1.0


def _build_shap_path(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_val_df: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
    shap_step: int,
    shap_sample_ratio: float,
    shap_max_samples: int,
    xgb_jobs: int,
    use_gpu: bool,
    progress_callback=None,
) -> list[dict]:
    current = list(X_train_df.columns)
    path = []
    step = max(1, int(shap_step))
    rng = np.random.default_rng(seed)

    total_steps = int(np.ceil((len(current) - 1) / step)) + 1
    step_idx = 0
    while len(current) >= 1:
        step_idx += 1
        if progress_callback is not None:
            progress_callback(step_idx, total_steps)
        t0 = time.perf_counter()
        model = make_model(seed, xgb_jobs, use_gpu)
        model.fit(X_train_df[current], y_train)
        fit_time_s = time.perf_counter() - t0

        pred = model.predict(X_val_df[current])
        rmse = float(root_mean_squared_error(y_val, pred))
        mae = float(mean_absolute_error(y_val, pred))
        mape = _safe_mape(y_val, pred)
        mse = float(mean_squared_error(y_val, pred))

        n_val = X_val_df.shape[0]
        n_shap = min(shap_max_samples, max(50, int(round(n_val * shap_sample_ratio))))
        n_shap = min(n_shap, n_val)
        idx = rng.choice(n_val, size=n_shap, replace=False)
        X_shap = X_val_df[current].iloc[idx]

        t1 = time.perf_counter()
        try:
            dmat = DMatrix(X_shap, feature_names=current, enable_categorical=True)
            contribs = model.get_booster().predict(dmat, pred_contribs=True)
            shap_vals = contribs[:, :-1]
            scores = np.mean(np.abs(shap_vals), axis=0)
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "XGBoost pred_contribs failed at n_features=%d (seed=%d). Falling back to feature_importances_. Error: %s",
                len(current),
                seed,
                exc,
            )
            scores = model.feature_importances_
        shap_time_s = time.perf_counter() - t1

        order = np.argsort(scores)[::-1]
        ranked = [current[i] for i in order]
        ranked_scores = [float(scores[i]) for i in order]

        drop_n = min(step, max(1, len(current) - 1)) if len(current) > 1 else 0
        dropped = ranked[-drop_n:] if drop_n > 0 else []

        path.append(
            {
                "n_features": len(current),
                "features": current.copy(),
                "ranked_features": ranked,
                "ranked_scores": ranked_scores,
                "dropped_features": dropped,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "mse": mse,
                "fit_time_s": fit_time_s,
                "shap_time_s": shap_time_s,
            }
        )

        if len(current) == 1:
            break
        current = [f for f in current if f not in set(dropped)]

    return path


def _select_from_path(path: list[dict], k_keep: int) -> tuple[list[str], dict]:
    if not path:
        raise ValueError("Selection path is empty; cannot select features")
    eligible = [s for s in path if s["n_features"] >= k_keep]
    if not eligible:
        snap = path[-1]
        return snap["features"], snap
    snap = min(eligible, key=lambda s: s["n_features"])
    return snap["features"][:k_keep], snap


def _run_strategy_job(
    seed: int,
    fold: int,
    folds: int,
    strategy: str,
    k_levels: list[float],
    done: set[tuple],
    arff_path: str,
    target_col: str | None,
    shap_step: int,
    shap_sample_ratio: float,
    shap_max_samples: int,
    xgb_jobs: int,
    use_gpu: bool,
    raw_path_str: str,
    sel_path_str: str,
    path_out_str: str,
    lock_raw_str: str,
    lock_sel_str: str,
    lock_path_str: str,
) -> tuple[int, int]:
    X_df, y, feature_names = _load_allstate_arff(arff_path, target_col)

    rows: list[dict] = []
    sels: list[dict] = []
    paths_rows: list[dict] = []
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(kf.split(X_df))
    tr, te = splits[fold - 1]
    X_tr_df = X_df.iloc[tr].copy()
    X_te_df = X_df.iloc[te].copy()
    y_tr = y[tr]
    y_te = y[te]

    n_feat = len(feature_names)
    fi_model = make_model(seed, xgb_jobs, use_gpu)
    fi_model.fit(X_tr_df, y_tr)
    fi_order = np.argsort(fi_model.feature_importances_)[::-1]
    fi_rank = [feature_names[i] for i in fi_order]

    strat_label = "fi" if strategy == "native_fi" else ("shaprfe" if strategy == "custom_shap_rfe" else "hybrid")
    full_path: list[dict] = []
    valid_k_levels = [k for k in k_levels if _is_valid_k_for_strategy(strategy, float(k))]

    if strategy == "native_fi":
        print(f"repeat: {seed} | fold: {fold}/{folds} | strat: {strat_label} | progress: n/a", flush=True)
    elif strategy == "custom_shap_rfe":
        full_path = _build_shap_path(
            X_tr_df,
            y_tr,
            X_te_df,
            y_te,
            seed,
            shap_step,
            shap_sample_ratio,
            shap_max_samples,
            xgb_jobs,
            use_gpu,
            progress_callback=lambda i, t: print(f"repeat: {seed} | fold: {fold}/{folds} | strat: shaprfe | progress: {i}/{t}", flush=True),
        )
    else:
        print(f"repeat: {seed} | fold: {fold}/{folds} | strat: hybrid | progress: 0/{len(valid_k_levels)}", flush=True)

    path_cache: dict[tuple[str, ...], list[dict]] = {}
    hybrid_k_done = 0
    strategy_has_work = False
    for k in valid_k_levels:
        key = (strategy, float(k), fold, seed)
        if key in done:
            if strategy == "hybrid_fi_custom_shap_rfe":
                hybrid_k_done += 1
                print(f"repeat: {seed} | fold: {fold}/{folds} | strat: hybrid | progress: {hybrid_k_done}/{len(valid_k_levels)}", flush=True)
            continue
        strategy_has_work = True

        k_keep = max(1, int(round(n_feat * k)))
        sel_t0 = time.perf_counter()
        if strategy == "native_fi":
            selected = fi_rank[:k_keep]
            stage_fi_keep = selected.copy()
            stage_shap_keep = selected.copy()
        elif strategy == "custom_shap_rfe":
            selected, _ = _select_from_path(full_path, k_keep)
            stage_fi_keep = feature_names.copy()
            stage_shap_keep = selected.copy()
        else:
            n_drop = n_feat - k_keep
            drop_fi = int(np.ceil(n_drop / 2.0))
            if drop_fi <= 0:
                keep_after_fi = fi_rank.copy()
            else:
                keep_after_fi = [f for f in fi_rank if f not in set(fi_rank[-drop_fi:])]
            if not keep_after_fi:
                keep_after_fi = [fi_rank[0]]
            pool_key = tuple(keep_after_fi)
            if pool_key not in path_cache:
                path_cache[pool_key] = _build_shap_path(
                    X_tr_df[keep_after_fi],
                    y_tr,
                    X_te_df[keep_after_fi],
                    y_te,
                    seed,
                    shap_step,
                    shap_sample_ratio,
                    shap_max_samples,
                    xgb_jobs,
                    use_gpu,
                    progress_callback=None,
                )
            k_keep_hybrid = min(k_keep, len(keep_after_fi))
            selected, _ = _select_from_path(path_cache[pool_key], k_keep_hybrid)
            stage_fi_keep = keep_after_fi
            stage_shap_keep = selected.copy()
            hybrid_k_done += 1
            print(f"repeat: {seed} | fold: {fold}/{folds} | strat: hybrid | progress: {hybrid_k_done}/{len(valid_k_levels)}", flush=True)

        selection_time_s = time.perf_counter() - sel_t0
        model = make_model(seed, xgb_jobs, use_gpu)
        tfit = time.perf_counter()
        model.fit(X_tr_df[selected], y_tr)
        fit_time_s = time.perf_counter() - tfit
        tpred = time.perf_counter()
        pred = model.predict(X_te_df[selected])
        pred_time_s = time.perf_counter() - tpred

        strategy_out = "baseline" if (strategy == "native_fi" and float(k) == 1.0) else strategy

        rows.append({
            "dataset": "Allstate",
            "model": "xgboost",
            "strategy": strategy_out,
            "k_pct": float(k),
            "fold": fold,
            "repeat_seed": seed,
            "rmse": float(root_mean_squared_error(y_te, pred)),
            "mae": float(mean_absolute_error(y_te, pred)),
            "mape": _safe_mape(y_te, pred),
            "mse": float(mean_squared_error(y_te, pred)),
            "selection_time_s": float(selection_time_s),
            "train_time_s": float(fit_time_s),
            "predict_time_s": float(pred_time_s),
            "total_time_s": float(selection_time_s + fit_time_s + pred_time_s),
        })
        sels.append({
            "dataset": "Allstate",
            "model": "xgboost",
            "strategy": strategy_out,
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
        })

    if strategy == "custom_shap_rfe":
        for snap in full_path:
            paths_rows.append(
                {
                    "dataset": "Allstate",
                    "model": "xgboost",
                    "strategy": "custom_shap_rfe",
                    "fold": fold,
                    "repeat_seed": seed,
                    "n_features": snap["n_features"],
                    "features": "|".join(snap["features"]),
                    "ranked_features": "|".join(snap["ranked_features"]),
                    "ranked_scores": "|".join([f"{v:.12g}" for v in snap["ranked_scores"]]),
                    "dropped_features": "|".join(snap["dropped_features"]),
                    "rmse": snap["rmse"],
                    "mae": snap["mae"],
                    "mape": snap["mape"],
                    "mse": snap["mse"],
                    "fit_time_s": snap["fit_time_s"],
                    "shap_time_s": snap["shap_time_s"],
                }
            )

    _append_csv_locked(pd.DataFrame(rows), Path(raw_path_str), Path(lock_raw_str))
    _append_csv_locked(pd.DataFrame(sels), Path(sel_path_str), Path(lock_sel_str))
    _append_csv_locked(pd.DataFrame(paths_rows), Path(path_out_str), Path(lock_path_str))
    return (1 if strategy_has_work else 0), len(rows)


def run_experiments(pmap: dict[str, Path], quick: bool, resume: bool, arff_path: str, target_col: str | None, workers: int, shap_step: int, shap_sample_ratio: float, shap_max_samples: int, xgb_jobs: int, use_gpu: bool) -> None:
    repeats = [42, 123] if quick else [42, 123, 256, 512, 999]
    folds = 2 if quick else 10
    k_levels = [0.5, 1.0] if quick else [0.05, 0.10, 0.15, 0.25, 0.50, 1.0]

    raw_path = pmap["outputs"] / "exp3_results_raw.csv"
    sel_path = pmap["outputs"] / "exp3_selections_raw.csv"
    path_path = pmap["outputs"] / "exp3_paths_raw.csv"

    existing = pd.read_csv(raw_path) if (resume and raw_path.exists()) else pd.DataFrame()
    done = set()
    if not existing.empty:
        for _, r in existing.iterrows():
            s = str(r["strategy"])
            if s == "baseline":
                s = "native_fi"
            done.add((s, float(r["k_pct"]), int(r["fold"]), int(r["repeat_seed"])))

    total_experiments = len(repeats) * folds * 3
    remaining_experiments = 0
    for seed in repeats:
        for fold in range(1, folds + 1):
            for strategy in ["native_fi", "custom_shap_rfe", "hybrid_fi_custom_shap_rfe"]:
                valid_k_levels = [k for k in k_levels if _is_valid_k_for_strategy(strategy, float(k))]
                if any((strategy, float(k), fold, seed) not in done for k in valid_k_levels):
                    remaining_experiments += 1

    X_check, _, feat_check = _load_allstate_arff(arff_path, target_col)
    cat_cols = int(sum(str(X_check[c].dtype) == "category" for c in X_check.columns))
    logging.info("Experiment 3 scope locked: dataset=Allstate ARFF, model=xgboost")
    logging.info("Loaded Allstate rows=%d features=%d categorical=%d", X_check.shape[0], len(feat_check), cat_cols)
    logging.info("k levels: %s", ", ".join([str(x) for x in k_levels]))
    logging.info("repeats=%s folds=%d workers=%d", repeats, folds, workers)
    logging.info("custom SHAP-RFE config: step=%d sample_ratio=%.3f max_samples=%d", shap_step, shap_sample_ratio, shap_max_samples)
    logging.info("xgboost threads per fit: %d", xgb_jobs)
    logging.info("xgboost device: %s", "cuda" if use_gpu else "cpu")
    logging.info("planned experiments=%d | remaining experiments=%d", total_experiments, remaining_experiments)

    pbar = tqdm(total=remaining_experiments, desc="Experiment2 repeat-fold-strategy", unit="exp")

    tasks = [(seed, fold, strategy) for seed in repeats for fold in range(1, folds + 1) for strategy in ["native_fi", "custom_shap_rfe", "hybrid_fi_custom_shap_rfe"]]
    lock_raw = pmap["outputs"] / "exp3_results_raw.csv.lock"
    lock_sel = pmap["outputs"] / "exp3_selections_raw.csv.lock"
    lock_path = pmap["outputs"] / "exp3_paths_raw.csv.lock"

    total_rows = 0

    if workers <= 1:
        for seed, fold, strategy in tasks:
            exp_done, rows_written = _run_strategy_job(
                seed,
                fold,
                folds,
                strategy,
                k_levels,
                done,
                arff_path,
                target_col,
                shap_step,
                shap_sample_ratio,
                shap_max_samples,
                xgb_jobs,
                use_gpu,
                str(raw_path),
                str(sel_path),
                str(path_path),
                str(lock_raw),
                str(lock_sel),
                str(lock_path),
            )
            total_rows += rows_written
            pbar.update(exp_done)
            logging.info("job done seed=%d fold=%d strat=%s rows_written=%d total_rows=%d", seed, fold, strategy, rows_written, total_rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _run_strategy_job,
                    seed,
                    fold,
                    folds,
                    strategy,
                    k_levels,
                    done,
                    arff_path,
                    target_col,
                    shap_step,
                    shap_sample_ratio,
                    shap_max_samples,
                    xgb_jobs,
                    use_gpu,
                    str(raw_path),
                    str(sel_path),
                    str(path_path),
                    str(lock_raw),
                    str(lock_sel),
                    str(lock_path),
                )
                for seed, fold, strategy in tasks
            ]
            completed = 0
            for fut in as_completed(futs):
                exp_done, rows_written = fut.result()
                total_rows += rows_written
                pbar.update(exp_done)
                completed += 1
                logging.info("experiment jobs done %d/%d (+%d rows, total=%d)", completed, len(futs), rows_written, total_rows)

    pbar.close()


def run_summary(pmap: dict[str, Path]) -> None:
    raw = pd.read_csv(pmap["outputs"] / "exp3_results_raw.csv")
    # Canonicalize baseline semantics: only baseline at k=1.0
    raw = raw.copy()
    raw.loc[(raw["k_pct"].astype(float) == 1.0) & (raw["strategy"] == "native_fi"), "strategy"] = "baseline"
    raw = raw[~((raw["k_pct"].astype(float) == 1.0) & (~raw["strategy"].isin(["baseline"])))].copy()
    raw = raw[~((raw["k_pct"].astype(float) < 1.0) & (raw["strategy"] == "baseline"))].copy()
    s = raw.groupby(["dataset", "model", "strategy", "k_pct"], as_index=False).agg(
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"),
        mean_mape=("mape", "mean"),
        mean_selection_time_s=("selection_time_s", "mean"),
        mean_total_time_s=("total_time_s", "mean"),
    )
    s.to_csv(pmap["outputs"] / "exp3_results_summary.csv", index=False)


def run_stability(pmap: dict[str, Path]) -> None:
    sel = pd.read_csv(pmap["outputs"] / "exp3_selections_raw.csv")
    rows = []
    for (strategy, k), g in sel.groupby(["strategy", "k_pct"]):
        sets = [set(v.split("|")) for v in g["selected_features"].tolist()]
        votes = {}
        for s in sets:
            for f in s:
                votes[f] = votes.get(f, 0) + 1
        maj = {f for f, c in votes.items() if c >= len(sets) / 2}
        rows.append({"strategy": strategy, "k_pct": float(k), "mean_jaccard_to_majority": float(np.mean([_jaccard(s, maj) for s in sets]))})
    pd.DataFrame(rows).to_csv(pmap["outputs"] / "exp3_stability_analysis.csv", index=False)


def run_multicollinearity(pmap: dict[str, Path], arff_path: str, target_col: str | None) -> None:
    X, _, _ = _load_allstate_arff(arff_path, target_col)
    sel = pd.read_csv(pmap["outputs"] / "exp3_selections_raw.csv")
    out = []
    for (strategy, k), g in sel.groupby(["strategy", "k_pct"]):
        means = []
        highs = []
        for feats in g["selected_features"]:
            cols = feats.split("|")
            num = X[cols].select_dtypes(include=[np.number])
            if num.shape[1] <= 1:
                means.append(0.0)
                highs.append(0.0)
                continue
            c = num.corr().abs().to_numpy()
            if c.shape[0] <= 1:
                means.append(0.0)
                highs.append(0.0)
                continue
            tri = c[np.triu_indices(c.shape[0], k=1)]
            means.append(float(np.mean(tri)))
            highs.append(float(np.mean(tri >= 0.8)))
        out.append({"strategy": strategy, "k_pct": float(k), "mean_abs_pairwise_corr": float(np.mean(means)), "mean_high_corr_pair_ratio_r_ge_0_8": float(np.mean(highs))})
    pd.DataFrame(out).to_csv(pmap["outputs"] / "exp3_multicollinearity_analysis.csv", index=False)


def run_figures(pmap: dict[str, Path], arff_path: str, target_col: str | None) -> None:
    sns.set_theme(style="whitegrid")
    fig_dir = pmap["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Always refresh derived tables before plotting
    run_summary(pmap)
    run_stability(pmap)
    run_multicollinearity(pmap, arff_path, target_col)

    raw = pd.read_csv(pmap["outputs"] / "exp3_results_raw.csv")
    summary = pd.read_csv(pmap["outputs"] / "exp3_results_summary.csv")

    raw = raw.copy()
    raw.loc[(raw["k_pct"].astype(float) == 1.0) & (raw["strategy"] == "native_fi"), "strategy"] = "baseline"
    raw = raw[~((raw["k_pct"].astype(float) == 1.0) & (~raw["strategy"].isin(["baseline"])))].copy()
    raw = raw[~((raw["k_pct"].astype(float) < 1.0) & (raw["strategy"] == "baseline"))].copy()

    stab_path = pmap["outputs"] / "exp3_stability_analysis.csv"
    mc_path = pmap["outputs"] / "exp3_multicollinearity_analysis.csv"

    k_order_num = sorted(summary["k_pct"].unique(), reverse=True)
    k_order = [str(int(k * 100)) for k in k_order_num]
    hue_order = ["native_fi", "custom_shap_rfe", "hybrid_fi_custom_shap_rfe", "baseline"]

    fig, ax = plt.subplots(figsize=(9, 5))
    tmp = summary.copy()
    tmp["k_label"] = (tmp["k_pct"] * 100).astype(int).astype(str)
    sns.barplot(data=tmp, x="k_label", y="mean_rmse", hue="strategy", hue_order=hue_order, order=k_order, ax=ax)
    ax.set_title("RMSE by Strategy and k")
    ax.set_xlabel("k% features kept (descending)")
    ax.set_ylabel("Mean RMSE")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp3_rmse_by_strategy_k.png", dpi=300)
    plt.close(fig)

    # RMSE distribution (boxplot) by strategy at each k
    fig, ax = plt.subplots(figsize=(10, 5))
    rb = raw.copy()
    rb["k_label"] = (rb["k_pct"] * 100).astype(int).astype(str)
    sns.boxplot(data=rb, x="k_label", y="rmse", hue="strategy", hue_order=hue_order, order=k_order, ax=ax)
    ax.set_title("RMSE Distribution by Strategy and k")
    ax.set_xlabel("k% features kept (descending)")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp3_rmse_boxplot_by_strategy_k.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    t = raw[raw["k_pct"].astype(float) < 1.0].groupby("strategy", as_index=False).agg(selection=("selection_time_s", "mean"), total=("total_time_s", "mean")).melt(id_vars=["strategy"], var_name="time_type", value_name="seconds")
    sns.barplot(data=t, x="strategy", y="seconds", hue="time_type", ax=ax)
    ax.set_title("Execution Time by Strategy")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp3_time_by_strategy.png", dpi=300)
    plt.close(fig)

    if stab_path.exists():
        stab = pd.read_csv(stab_path)
        if not stab.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            sp = stab.copy()
            sp["k_label"] = (sp["k_pct"] * 100).astype(int).astype(str)
            sns.barplot(data=sp, x="k_label", y="mean_jaccard_to_majority", hue="strategy", hue_order=hue_order, order=k_order, ax=ax)
            ax.set_title("Stability (Jaccard to Majority) by Strategy and k")
            ax.set_xlabel("k% features kept (descending)")
            ax.set_ylabel("Mean Jaccard")
            fig.tight_layout()
            fig.savefig(fig_dir / "exp3_stability_by_strategy_k.png", dpi=300)
            plt.close(fig)

    if mc_path.exists():
        mc = pd.read_csv(mc_path)
        if not mc.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            mp = mc.copy()
            mp["k_label"] = (mp["k_pct"] * 100).astype(int).astype(str)
            sns.barplot(data=mp, x="k_label", y="mean_abs_pairwise_corr", hue="strategy", hue_order=hue_order, order=k_order, ax=axes[0])
            axes[0].set_title("Mean Absolute Pairwise Correlation")
            axes[0].set_xlabel("k% (descending)")
            axes[0].set_ylabel("Mean |corr|")
            sns.barplot(data=mp, x="k_label", y="mean_high_corr_pair_ratio_r_ge_0_8", hue="strategy", hue_order=hue_order, order=k_order, ax=axes[1])
            axes[1].set_title("High-Correlation Pair Ratio (|r|>=0.8)")
            axes[1].set_xlabel("k% (descending)")
            axes[1].set_ylabel("Ratio")
            lg = axes[0].get_legend()
            if lg is not None:
                lg.remove()
            fig.tight_layout()
            fig.savefig(fig_dir / "exp3_multicollinearity_by_strategy_k.png", dpi=300)
            plt.close(fig)

    # Accuracy vs compute trade-off
    fig, ax = plt.subplots(figsize=(8, 6))
    trade = summary.groupby("strategy", as_index=False).agg(mean_rmse=("mean_rmse", "mean"), mean_total_time_s=("mean_total_time_s", "mean"))
    sns.scatterplot(data=trade, x="mean_total_time_s", y="mean_rmse", hue="strategy", s=120, ax=ax)
    for _, r in trade.iterrows():
        ax.text(float(r["mean_total_time_s"]), float(r["mean_rmse"]), f" {r['strategy']}", va="center")
    ax.set_title("RMSE vs Total Time (Strategy Means)")
    ax.set_xlabel("Mean Total Time (s)")
    ax.set_ylabel("Mean RMSE")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp3_rmse_vs_time_tradeoff.png", dpi=300)
    plt.close(fig)


def run_analysis_markdown(pmap: dict[str, Path]) -> None:
    raw = pd.read_csv(pmap["outputs"] / "exp3_results_raw.csv")
    summary = pd.read_csv(pmap["outputs"] / "exp3_results_summary.csv")
    best = summary.loc[summary.groupby("k_pct")["mean_rmse"].idxmin(), ["k_pct", "strategy", "mean_rmse", "mean_mae", "mean_mape"]]
    lines = [
        "# Experiment 3 Results Draft",
        "",
        "- Dataset: Allstate",
        "- Model: XGBoost",
        f"- Raw observations: {len(raw)}",
        "",
        "## Best by k (RMSE)",
        best.to_markdown(index=False),
    ]
    (pmap["outputs"] / "exp3_results_draft.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging(True)
    random.seed(SEED)
    np.random.seed(SEED)

    pmap = paths(Path.cwd(), run_id=args.run_id)
    pmap["outputs"].mkdir(parents=True, exist_ok=True)
    pmap["figures"].mkdir(parents=True, exist_ok=True)
    pmap["logs"].mkdir(parents=True, exist_ok=True)

    logging.info("Experiment 3 run root: %s", pmap["root"])
    logging.info("Experiment 3 is fixed to dataset=Allstate ARFF and model=xgboost")
    if args.workers > 1 and args.xgb_jobs > 1:
        logging.warning(
            "High contention config detected (workers=%d, xgb_jobs=%d). Consider xgb_jobs=1 for stable throughput.",
            args.workers,
            args.xgb_jobs,
        )

    if args.step in ["all", "experiments"]:
        run_experiments(
            pmap,
            quick=args.quick,
            resume=args.resume,
            arff_path=args.arff_path,
            target_col=args.target_col,
            workers=max(1, args.workers),
            shap_step=max(1, args.shap_step),
            shap_sample_ratio=float(args.shap_sample_ratio),
            shap_max_samples=max(50, int(args.shap_max_samples)),
            xgb_jobs=max(1, int(args.xgb_jobs)),
            use_gpu=not args.no_gpu,
        )
    if args.step in ["all", "summary"]:
        run_summary(pmap)
    if args.step in ["all", "stability"]:
        run_stability(pmap)
    if args.step in ["all", "multicollinearity"]:
        run_multicollinearity(pmap, args.arff_path, args.target_col)
    if args.step in ["all", "figures"]:
        run_figures(pmap, args.arff_path, args.target_col)
    if args.step in ["all", "analysis"]:
        run_analysis_markdown(pmap)


if __name__ == "__main__":
    main()
