from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager
import queue as pyqueue
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import DMatrix, XGBRegressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel timing test for custom SHAP-RFE")
    p.add_argument("--use-preprocessed-dir", default="pre_study/data/processed")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--xgb-jobs", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--fold", type=int, default=1, help="1-based fold index")
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--shap-step", type=int, default=2)
    p.add_argument("--shap-sample-ratio", type=float, default=0.10)
    p.add_argument("--shap-max-samples", type=int, default=1600)
    p.add_argument("--min-k", type=float, default=0.05)
    p.add_argument("--no-gpu", action="store_true")
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
    }
    if use_gpu:
        params["device"] = "cuda"
    return XGBRegressor(**params)


def load_superconductor(processed_dir: str) -> tuple[pd.DataFrame, np.ndarray]:
    path = f"{processed_dir}/superconductor_processed.csv"
    df = pd.read_csv(path)
    y = pd.to_numeric(df["critical_temp"], errors="coerce")
    X = df.drop(columns=["critical_temp"]).apply(pd.to_numeric, errors="coerce")
    clean = X.copy()
    clean["target"] = y
    clean = clean.dropna()
    y_clean = clean.pop("target").to_numpy()
    return clean, y_clean


def run_one(
    seed: int,
    X_df: pd.DataFrame,
    y: np.ndarray,
    fold: int,
    n_splits: int,
    shap_step: int,
    shap_sample_ratio: float,
    shap_max_samples: int,
    min_k: float,
    xgb_jobs: int,
    use_gpu: bool,
    progress_queue,
) -> dict:
    feature_names = list(X_df.columns)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(kf.split(X_df))
    tr, te = splits[fold - 1]

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
    n_min = max(1, int(round(n_total * min_k)))
    current = feature_names.copy()
    step = max(1, int(shap_step))
    rng = np.random.default_rng(seed)
    total_iters = int(np.ceil((n_total - n_min) / step)) + 1
    progress_queue.put((seed, "init", total_iters))

    iters = 0
    model = None
    t0 = time.perf_counter()
    while len(current) >= n_min:
        iters += 1
        progress_queue.put((seed, "tick", iters))
        model = make_model(seed, xgb_jobs, use_gpu)
        model.fit(X_tr_df[current], y_tr)

        n_val = X_te_df.shape[0]
        n_shap = min(shap_max_samples, max(50, int(round(n_val * shap_sample_ratio))))
        n_shap = min(n_shap, n_val)
        idx = rng.choice(n_val, size=n_shap, replace=False)
        X_shap = X_te_df[current].iloc[idx]

        dmat = DMatrix(X_shap.to_numpy(), feature_names=current)
        contribs = model.get_booster().predict(dmat, pred_contribs=True)
        scores = np.mean(np.abs(contribs[:, :-1]), axis=0)
        order = np.argsort(scores)[::-1]
        ranked = [current[i] for i in order]

        if len(current) == n_min:
            break
        drop_n = min(step, len(current) - n_min)
        current = ranked[:-drop_n]

    elapsed = time.perf_counter() - t0
    progress_queue.put((seed, "done", iters))
    if model is None:
        raise RuntimeError("Benchmark loop did not run")
    pred = model.predict(X_te_df[current])
    rmse = float(np.sqrt(np.mean((y_te - pred) ** 2)))
    return {"seed": seed, "iters": iters, "seconds": elapsed, "rmse": rmse}


def main() -> None:
    args = parse_args()
    X_df, y = load_superconductor(args.use_preprocessed_dir)
    seeds = [42, 123, 256, 512, 999][: max(1, args.repeats)]

    t0 = time.perf_counter()
    results = []
    with Manager() as mgr:
        progress_q = mgr.Queue()
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [
                ex.submit(
                    run_one,
                    seed,
                    X_df,
                    y,
                    args.fold,
                    args.n_splits,
                    args.shap_step,
                    args.shap_sample_ratio,
                    args.shap_max_samples,
                    args.min_k,
                    args.xgb_jobs,
                    (not args.no_gpu),
                    progress_q,
                )
                for seed in seeds
            ]

            top = tqdm(total=len(futs), desc="Parallel seeds", unit="seed", position=0)
            bars: dict[int, tqdm] = {}
            pending = set(futs)

            while pending:
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for fut in done:
                    results.append(fut.result())
                    top.update(1)

                while True:
                    try:
                        seed, kind, value = progress_q.get_nowait()
                    except pyqueue.Empty:
                        break
                    if kind == "init":
                        if seed not in bars:
                            pos = len(bars) + 1
                            bars[seed] = tqdm(total=int(value), desc=f"seed {seed}", unit="iter", position=pos, leave=True)
                    elif kind == "tick" and seed in bars:
                        bars[seed].n = min(int(value), bars[seed].total)
                        bars[seed].refresh()
                    elif kind == "done" and seed in bars:
                        bars[seed].n = bars[seed].total
                        bars[seed].refresh()

            top.close()
            for b in bars.values():
                b.close()

    total = time.perf_counter() - t0
    df = pd.DataFrame(results).sort_values("seed")
    print(df.to_string(index=False))
    print(f"\nTotal wall time: {total:.2f}s")
    print(f"Mean seed time: {df['seconds'].mean():.2f}s")


if __name__ == "__main__":
    main()
