from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import DMatrix, XGBRegressor

from src.config import SEED, paths
from src.data_loading import load_all_datasets_from_local
from src.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 2: Superconductor + XGBoost + custom SHAP-RFE")
    p.add_argument("--step", choices=["all", "experiments", "summary", "stability", "multicollinearity", "figures", "analysis"], default="all")
    p.add_argument("--run-id", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--use-preprocessed-dir", default="pre_study/data/processed")
    p.add_argument("--shap-step", type=int, default=5)
    p.add_argument("--shap-sample-ratio", type=float, default=0.10)
    p.add_argument("--shap-max-samples", type=int, default=1600)
    return p.parse_args()


def make_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
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


def _checkpoint_write(existing: pd.DataFrame, rows: list[dict], sels: list[dict], paths_rows: list[dict], raw_path: Path, sel_path: Path, path_path: Path) -> None:
    merged = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if rows else existing
    merged.to_csv(raw_path, index=False)
    pd.DataFrame(sels).to_csv(sel_path, index=False)
    pd.DataFrame(paths_rows).to_csv(path_path, index=False)


def _build_shap_path(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_val_df: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
    shap_step: int,
    shap_sample_ratio: float,
    shap_max_samples: int,
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
        model = make_model(seed)
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
            dmat = DMatrix(X_shap.to_numpy(), feature_names=current)
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
    eligible = [s for s in path if s["n_features"] >= k_keep]
    if not eligible:
        snap = path[-1]
        return snap["features"], snap
    snap = min(eligible, key=lambda s: s["n_features"])
    return snap["features"][:k_keep], snap


def _run_seed_task(seed: int, folds: int, k_levels: list[float], done: set[tuple], preprocessed_dir: str, shap_step: int, shap_sample_ratio: float, shap_max_samples: int) -> tuple[list[dict], list[dict], list[dict], int]:
    ds_all = load_all_datasets_from_local(preprocessed_dir)
    ds = [d for d in ds_all if d["name"] == "Superconductor"]
    if not ds:
        raise RuntimeError("Superconductor not found")
    X_df = ds[0]["X"].copy()
    y = ds[0]["y"].to_numpy()
    feature_names = list(X_df.columns)

    rows: list[dict] = []
    sels: list[dict] = []
    paths_rows: list[dict] = []
    exp_done = 0
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (tr, te) in enumerate(kf.split(X_df), 1):
        print(f"repeat_seed={seed} | fold={fold} | start", flush=True)
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
        fi_model = make_model(seed)
        fi_model.fit(X_tr, y_tr)
        fi_order = np.argsort(fi_model.feature_importances_)[::-1]
        fi_rank = [feature_names[i] for i in fi_order]

        print(f"repeat_seed={seed} | fold={fold} | custom_shap_rfe | build full path", flush=True)
        full_path = _build_shap_path(
            X_tr_df,
            y_tr,
            X_te_df,
            y_te,
            seed,
            shap_step,
            shap_sample_ratio,
            shap_max_samples,
            progress_callback=lambda i, t: print(
                f"repeat: {seed} | fold: {fold}/{folds} | strat: shaprfe | progress: {i}/{t}",
                flush=True,
            ),
        )
        path_cache: dict[tuple[str, ...], list[dict]] = {}

        for strategy in ["native_fi", "custom_shap_rfe", "hybrid_fi_custom_shap_rfe"]:
            strat_label = "fi" if strategy == "native_fi" else ("shaprfe" if strategy == "custom_shap_rfe" else "hybrid")
            if strategy == "native_fi":
                print(f"repeat: {seed} | fold: {fold}/{folds} | strat: {strat_label} | progress: n/a", flush=True)
            elif strategy == "hybrid_fi_custom_shap_rfe":
                print(f"repeat: {seed} | fold: {fold}/{folds} | strat: {strat_label} | progress: 0/{len(k_levels)}", flush=True)

            strategy_has_work = False
            hybrid_k_done = 0
            for k in k_levels:
                key = (strategy, float(k), fold, seed)
                if key in done:
                    if strategy == "hybrid_fi_custom_shap_rfe":
                        hybrid_k_done += 1
                        print(
                            f"repeat: {seed} | fold: {fold}/{folds} | strat: hybrid | progress: {hybrid_k_done}/{len(k_levels)}",
                            flush=True,
                        )
                    continue

                strategy_has_work = True

                k_keep = max(1, int(round(n_feat * k)))
                sel_t0 = time.perf_counter()

                if strategy == "native_fi":
                    selected = fi_rank[:k_keep]
                    stage_fi_keep = selected.copy()
                    stage_shap_keep = selected.copy()
                    snap = None
                elif strategy == "custom_shap_rfe":
                    selected, snap = _select_from_path(full_path, k_keep)
                    stage_fi_keep = feature_names.copy()
                    stage_shap_keep = selected.copy()
                else:
                    n_drop = n_feat - k_keep
                    drop_fi = int(np.ceil(n_drop / 2.0))
                    keep_after_fi = [f for f in fi_rank if f not in set(fi_rank[-drop_fi:])]
                    pool_key = tuple(keep_after_fi)
                    if pool_key not in path_cache:
                        print(f"repeat_seed={seed} | fold={fold} | hybrid path build | pool={len(keep_after_fi)}", flush=True)
                        path_cache[pool_key] = _build_shap_path(
                            X_tr_df[keep_after_fi],
                            y_tr,
                            X_te_df[keep_after_fi],
                            y_te,
                            seed,
                            shap_step,
                            shap_sample_ratio,
                            shap_max_samples,
                            progress_callback=None,
                        )
                    selected, snap = _select_from_path(path_cache[pool_key], k_keep)
                    stage_fi_keep = keep_after_fi
                    stage_shap_keep = selected.copy()
                    hybrid_k_done += 1
                    print(
                        f"repeat: {seed} | fold: {fold}/{folds} | strat: hybrid | progress: {hybrid_k_done}/{len(k_levels)}",
                        flush=True,
                    )

                selection_time_s = time.perf_counter() - sel_t0
                model = make_model(seed)
                tfit = time.perf_counter()
                model.fit(X_tr_df[selected], y_tr)
                fit_time_s = time.perf_counter() - tfit
                tpred = time.perf_counter()
                pred = model.predict(X_te_df[selected])
                pred_time_s = time.perf_counter() - tpred

                rows.append(
                    {
                        "dataset": "Superconductor",
                        "model": "xgboost",
                        "strategy": strategy,
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

            if strategy_has_work:
                exp_done += 1

        for snap in full_path:
            paths_rows.append(
                {
                    "dataset": "Superconductor",
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

        print(f"repeat_seed={seed} | fold={fold} | done", flush=True)

    return rows, sels, paths_rows, exp_done


def run_experiments(pmap: dict[str, Path], quick: bool, resume: bool, preprocessed_dir: str, workers: int, shap_step: int, shap_sample_ratio: float, shap_max_samples: int) -> None:
    repeats = [42, 123] if quick else [42, 123, 256, 512, 999]
    folds = 2 if quick else 10
    k_levels = [0.5, 1.0] if quick else [0.05, 0.10, 0.15, 0.25, 0.50, 1.0]

    raw_path = pmap["outputs"] / "exp2_results_raw.csv"
    sel_path = pmap["outputs"] / "exp2_selections_raw.csv"
    path_path = pmap["outputs"] / "exp2_paths_raw.csv"

    existing = pd.read_csv(raw_path) if (resume and raw_path.exists()) else pd.DataFrame()
    done = set()
    if not existing.empty:
        for _, r in existing.iterrows():
            done.add((r["strategy"], float(r["k_pct"]), int(r["fold"]), int(r["repeat_seed"])))

    total_experiments = len(repeats) * folds * 3
    remaining_experiments = 0
    for seed in repeats:
        for fold in range(1, folds + 1):
            for strategy in ["native_fi", "custom_shap_rfe", "hybrid_fi_custom_shap_rfe"]:
                if any((strategy, float(k), fold, seed) not in done for k in k_levels):
                    remaining_experiments += 1

    rows: list[dict] = []
    sels: list[dict] = []
    paths_rows: list[dict] = []

    logging.info("Experiment 2 scope locked: dataset=Superconductor, model=xgboost")
    logging.info("k levels: %s", ", ".join([str(x) for x in k_levels]))
    logging.info("repeats=%s folds=%d workers=%d", repeats, folds, workers)
    logging.info("custom SHAP-RFE config: step=%d sample_ratio=%.3f max_samples=%d", shap_step, shap_sample_ratio, shap_max_samples)
    logging.info("planned experiments=%d | remaining experiments=%d", total_experiments, remaining_experiments)

    pbar = tqdm(total=remaining_experiments, desc="Experiment2 repeat-fold-strategy", unit="exp")

    if workers <= 1:
        for seed in repeats:
            r, s, pth, exp_done = _run_seed_task(seed, folds, k_levels, done, preprocessed_dir, shap_step, shap_sample_ratio, shap_max_samples)
            rows.extend(r)
            sels.extend(s)
            paths_rows.extend(pth)
            pbar.update(exp_done)
            _checkpoint_write(existing, rows, sels, paths_rows, raw_path, sel_path, path_path)
            logging.info("seed done=%d accumulated rows=%d", seed, len(rows))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_run_seed_task, seed, folds, k_levels, done, preprocessed_dir, shap_step, shap_sample_ratio, shap_max_samples)
                for seed in repeats
            ]
            completed = 0
            for fut in as_completed(futs):
                r, s, pth, exp_done = fut.result()
                rows.extend(r)
                sels.extend(s)
                paths_rows.extend(pth)
                pbar.update(exp_done)
                _checkpoint_write(existing, rows, sels, paths_rows, raw_path, sel_path, path_path)
                completed += 1
                logging.info("seed tasks done %d/%d (+%d rows, total=%d)", completed, len(futs), len(r), len(rows))

    pbar.close()


def run_summary(pmap: dict[str, Path]) -> None:
    raw = pd.read_csv(pmap["outputs"] / "exp2_results_raw.csv")
    s = raw.groupby(["dataset", "model", "strategy", "k_pct"], as_index=False).agg(
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"),
        mean_mape=("mape", "mean"),
        mean_selection_time_s=("selection_time_s", "mean"),
        mean_total_time_s=("total_time_s", "mean"),
    )
    s.to_csv(pmap["outputs"] / "exp2_results_summary.csv", index=False)


def run_stability(pmap: dict[str, Path]) -> None:
    sel = pd.read_csv(pmap["outputs"] / "exp2_selections_raw.csv")
    rows = []
    for (strategy, k), g in sel.groupby(["strategy", "k_pct"]):
        sets = [set(v.split("|")) for v in g["selected_features"].tolist()]
        votes = {}
        for s in sets:
            for f in s:
                votes[f] = votes.get(f, 0) + 1
        maj = {f for f, c in votes.items() if c >= len(sets) / 2}
        rows.append({"strategy": strategy, "k_pct": float(k), "mean_jaccard_to_majority": float(np.mean([_jaccard(s, maj) for s in sets]))})
    pd.DataFrame(rows).to_csv(pmap["outputs"] / "exp2_stability_analysis.csv", index=False)


def run_multicollinearity(pmap: dict[str, Path], preprocessed_dir: str) -> None:
    ds = [d for d in load_all_datasets_from_local(preprocessed_dir) if d["name"] == "Superconductor"][0]
    X = ds["X"]
    sel = pd.read_csv(pmap["outputs"] / "exp2_selections_raw.csv")
    out = []
    for (strategy, k), g in sel.groupby(["strategy", "k_pct"]):
        means = []
        highs = []
        for feats in g["selected_features"]:
            cols = feats.split("|")
            c = X[cols].corr().abs().to_numpy()
            if c.shape[0] <= 1:
                means.append(0.0)
                highs.append(0.0)
                continue
            tri = c[np.triu_indices(c.shape[0], k=1)]
            means.append(float(np.mean(tri)))
            highs.append(float(np.mean(tri >= 0.8)))
        out.append({"strategy": strategy, "k_pct": float(k), "mean_abs_pairwise_corr": float(np.mean(means)), "mean_high_corr_pair_ratio_r_ge_0_8": float(np.mean(highs))})
    pd.DataFrame(out).to_csv(pmap["outputs"] / "exp2_multicollinearity_analysis.csv", index=False)


def run_figures(pmap: dict[str, Path]) -> None:
    sns.set_theme(style="whitegrid")
    fig_dir = pmap["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(pmap["outputs"] / "exp2_results_raw.csv")
    summary = pd.read_csv(pmap["outputs"] / "exp2_results_summary.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    tmp = summary.copy()
    tmp["k_label"] = (tmp["k_pct"] * 100).astype(int).astype(str)
    sns.barplot(data=tmp, x="k_label", y="mean_rmse", hue="strategy", ax=ax)
    ax.set_title("RMSE by Strategy and k")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp2_rmse_by_strategy_k.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    t = raw.groupby("strategy", as_index=False).agg(selection=("selection_time_s", "mean"), total=("total_time_s", "mean")).melt(id_vars=["strategy"], var_name="time_type", value_name="seconds")
    sns.barplot(data=t, x="strategy", y="seconds", hue="time_type", ax=ax)
    ax.set_title("Execution Time by Strategy")
    fig.tight_layout()
    fig.savefig(fig_dir / "exp2_time_by_strategy.png", dpi=300)
    plt.close(fig)


def run_analysis_markdown(pmap: dict[str, Path]) -> None:
    raw = pd.read_csv(pmap["outputs"] / "exp2_results_raw.csv")
    summary = pd.read_csv(pmap["outputs"] / "exp2_results_summary.csv")
    best = summary.loc[summary.groupby("k_pct")["mean_rmse"].idxmin(), ["k_pct", "strategy", "mean_rmse", "mean_mae", "mean_mape"]]
    lines = [
        "# Experiment 2 Results Draft",
        "",
        "- Dataset: Superconductor",
        "- Model: XGBoost",
        f"- Raw observations: {len(raw)}",
        "",
        "## Best by k (RMSE)",
        best.to_markdown(index=False),
    ]
    (pmap["outputs"] / "exp2_results_draft.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging(True)
    random.seed(SEED)
    np.random.seed(SEED)

    pmap = paths(Path.cwd(), run_id=args.run_id)
    pmap["outputs"].mkdir(parents=True, exist_ok=True)
    pmap["figures"].mkdir(parents=True, exist_ok=True)
    pmap["logs"].mkdir(parents=True, exist_ok=True)

    logging.info("Experiment 2 run root: %s", pmap["root"])
    logging.info("Experiment 2 is fixed to dataset=Superconductor and model=xgboost")

    if args.step in ["all", "experiments"]:
        run_experiments(
            pmap,
            quick=args.quick,
            resume=args.resume,
            preprocessed_dir=args.use_preprocessed_dir,
            workers=max(1, args.workers),
            shap_step=max(1, args.shap_step),
            shap_sample_ratio=float(args.shap_sample_ratio),
            shap_max_samples=max(50, int(args.shap_max_samples)),
        )
    if args.step in ["all", "summary"]:
        run_summary(pmap)
    if args.step in ["all", "stability"]:
        run_stability(pmap)
    if args.step in ["all", "multicollinearity"]:
        run_multicollinearity(pmap, args.use_preprocessed_dir)
    if args.step in ["all", "figures"]:
        run_figures(pmap)
    if args.step in ["all", "analysis"]:
        run_analysis_markdown(pmap)


if __name__ == "__main__":
    main()
