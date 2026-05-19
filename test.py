from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import DMatrix, XGBRegressor

from src.data_loading import load_all_datasets_from_local


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick timing benchmark for custom SHAP-RFE path")
    p.add_argument("--use-preprocessed-dir", default="pre_study/data/processed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=1, help="1-based fold index")
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--step", type=int, default=1, help="features dropped per iteration")
    p.add_argument("--shap-sample-ratio", type=float, default=0.10)
    p.add_argument("--shap-max-samples", type=int, default=1600)
    p.add_argument("--min-k", type=float, default=0.05, help="lower bound fraction, e.g. 0.05")
    p.add_argument("--estimate-tasks", type=int, default=50, help="for rough total-time estimate")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds_all = load_all_datasets_from_local(args.use_preprocessed_dir)
    ds = [d for d in ds_all if d["name"] == "Superconductor"][0]
    X_df = ds["X"].copy()
    y = ds["y"].to_numpy()
    feature_names = list(X_df.columns)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(kf.split(X_df))
    if not (1 <= args.fold <= len(splits)):
        raise ValueError(f"fold must be in [1,{len(splits)}]")
    tr, te = splits[args.fold - 1]

    X_tr_raw = X_df.iloc[tr].to_numpy()
    X_te_raw = X_df.iloc[te].to_numpy()
    y_tr = y[tr]
    y_te = y[te]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)
    X_tr_df = pd.DataFrame(X_tr, columns=feature_names)
    X_te_df = pd.DataFrame(X_te, columns=feature_names)

    n_total = len(feature_names)
    n_min = max(1, int(round(n_total * args.min_k)))

    current = feature_names.copy()
    step = max(1, args.step)
    rng = np.random.default_rng(args.seed)

    snapshots = []
    n_iters = int(np.ceil((n_total - n_min) / step)) + 1
    pbar = tqdm(total=n_iters, desc="Custom SHAP-RFE iterations")
    t0_all = time.perf_counter()

    while len(current) >= n_min:
        t_fit_0 = time.perf_counter()
        model = make_model(args.seed)
        model.fit(X_tr_df[current], y_tr)
        fit_time = time.perf_counter() - t_fit_0

        n_val = X_te_df.shape[0]
        n_shap = min(args.shap_max_samples, max(50, int(round(n_val * args.shap_sample_ratio))))
        idx = rng.choice(n_val, size=n_shap, replace=False)
        X_shap = X_te_df[current].iloc[idx]

        t_shap_0 = time.perf_counter()
        dmat = DMatrix(X_shap.to_numpy(), feature_names=current)
        contribs = model.get_booster().predict(dmat, pred_contribs=True)
        shap_vals = contribs[:, :-1]
        scores = np.mean(np.abs(shap_vals), axis=0)
        shap_time = time.perf_counter() - t_shap_0

        order = np.argsort(scores)[::-1]
        ranked = [current[i] for i in order]

        pred = model.predict(X_te_df[current])
        rmse = float(np.sqrt(np.mean((y_te - pred) ** 2)))

        snapshots.append(
            {
                "n_features": len(current),
                "fit_time_s": fit_time,
                "shap_time_s": shap_time,
                "rmse": rmse,
            }
        )

        pbar.set_postfix(features=len(current), fit_s=f"{fit_time:.2f}", shap_s=f"{shap_time:.2f}")
        pbar.update(1)

        if len(current) == n_min:
            break
        drop_n = min(step, len(current) - n_min)
        current = ranked[:-drop_n]

    pbar.close()
    total_s = time.perf_counter() - t0_all

    df = pd.DataFrame(snapshots)
    print("\nTiming summary")
    print(df.describe().to_string())
    print(f"\nOne fold path total time: {total_s:.2f} s")
    est = total_s * max(1, args.estimate_tasks)
    print(f"Estimated total for {args.estimate_tasks} similar tasks: {est/3600:.2f} hours")


if __name__ == "__main__":
    main()
