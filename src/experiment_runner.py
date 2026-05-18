from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.config import MODELS, STRATEGIES
from src.feature_selection import select_top_k_indices, strategy_rank
from src.models import make_model
from src.preprocessing import scale_train_test
from src.results_io import RAW_COLUMNS


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


def _key_tuple(row: dict[str, Any]) -> tuple:
    return (
        row["dataset"],
        row["model"],
        row["strategy"],
        float(row["k_pct"]),
        int(row["fold"]),
        int(row["repeat_seed"]),
    )


def build_done_keys(raw_df: pd.DataFrame) -> set[tuple]:
    done = set()
    if raw_df.empty:
        return done
    for _, row in raw_df.iterrows():
        done.add(_key_tuple(row.to_dict()))
    return done


@dataclass(frozen=True)
class ModelTask:
    repeat_seed: int
    repeat_idx: int
    repeat_total: int
    dataset_idx: int
    dataset_total: int
    dataset_name: str
    model_idx: int
    model_total: int
    model_name: str


def _run_model_task(task: ModelTask, ds: dict, run_config, done_keys: set[tuple]) -> tuple[list[dict], list[dict]]:
    random.seed(task.repeat_seed)
    np.random.seed(task.repeat_seed)

    X_df = ds["X"]
    y_series = ds["y"]
    feature_names = np.array(ds["feature_names"])

    results: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []

    print(f"repeat {task.repeat_idx}/{task.repeat_total} (seed={task.repeat_seed})")
    print(f"dataset {task.dataset_idx}/{task.dataset_total} ({task.dataset_name})")
    print(f"model {task.model_idx}/{task.model_total} ({task.model_name})")

    kf = KFold(n_splits=run_config.folds, shuffle=True, random_state=task.repeat_seed)
    folds = list(kf.split(X_df))

    for fold_num, (train_idx, test_idx) in enumerate(folds, start=1):
        print(f"fold {fold_num}/{run_config.folds}")
        X_train_raw = X_df.iloc[train_idx].to_numpy()
        X_test_raw = X_df.iloc[test_idx].to_numpy()
        y_train = y_series.iloc[train_idx].to_numpy()
        y_test = y_series.iloc[test_idx].to_numpy()

        X_train, X_test = scale_train_test(X_train_raw, X_test_raw)
        n_features = X_train.shape[1]
        baseline_metric_cache: dict[tuple, tuple[float, float, float]] = {}
        baseline_time_cache: dict[tuple, tuple[float, float, float]] = {}
        baseline_key = (task.dataset_name, task.model_name, 1.0, fold_num, task.repeat_seed)

        for strategy_idx, strategy in enumerate(STRATEGIES, start=1):
            print(f"strategy {strategy_idx}/{len(STRATEGIES)} ({strategy})")
            ranking = None
            selection_time_s = 0.0
            if any(k < 1.0 for k in run_config.k_levels):
                t0 = time.perf_counter()
                ranking = strategy_rank(
                    strategy=strategy,
                    X_train=X_train,
                    y_train=y_train,
                    seed=task.repeat_seed,
                    model_name=task.model_name,
                    shap_sample_ratio=run_config.shap_sample_ratio,
                    use_gpu=run_config.use_gpu,
                )
                selection_time_s = time.perf_counter() - t0

            for k_idx, k_pct in enumerate(run_config.k_levels, start=1):
                print(f"k {k_idx}/{len(run_config.k_levels)} ({int(k_pct * 100)}%)")
                row = {
                    "dataset": task.dataset_name,
                    "model": task.model_name,
                    "strategy": strategy,
                    "k_pct": float(k_pct),
                    "fold": fold_num,
                    "repeat_seed": task.repeat_seed,
                }
                key = _key_tuple(row)
                if key in done_keys:
                    continue

                if k_pct == 1.0:
                    if baseline_key not in baseline_metric_cache:
                        t_total_0 = time.perf_counter()
                        preds, train_time_s, predict_time_s = _fit_predict_timed(
                            task.model_name,
                            task.repeat_seed,
                            X_train,
                            y_train,
                            X_test,
                            run_config.use_gpu,
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
                    assert ranking is not None
                    selected_idx = select_top_k_indices(ranking, n_features, k_pct)
                    X_train_sel = X_train[:, selected_idx]
                    X_test_sel = X_test[:, selected_idx]

                    t_total_0 = time.perf_counter()
                    preds, train_time_s, predict_time_s = _fit_predict_timed(
                        task.model_name,
                        task.repeat_seed,
                        X_train_sel,
                        y_train,
                        X_test_sel,
                        run_config.use_gpu,
                    )
                    total_time_s = time.perf_counter() - t_total_0
                    mape = mean_absolute_percentage_error(y_test, preds)
                    mse = mean_squared_error(y_test, preds)
                    rmse = root_mean_squared_error(y_test, preds)
                    local_selection_time = selection_time_s

                if not np.isfinite(mape):
                    logging.warning(
                        "Skipping inf/invalid MAPE for dataset=%s model=%s strategy=%s k=%.2f fold=%d repeat=%d",
                        task.dataset_name,
                        task.model_name,
                        strategy,
                        k_pct,
                        fold_num,
                        task.repeat_seed,
                    )
                    continue

                results.append(
                    {
                        **row,
                        "mape": float(mape),
                        "mse": float(mse),
                        "rmse": float(rmse),
                        "selection_time_s": float(local_selection_time),
                        "train_time_s": float(train_time_s),
                        "predict_time_s": float(predict_time_s),
                        "total_time_s": float(total_time_s + local_selection_time),
                    }
                )
                selections.append(
                    {
                        "dataset": task.dataset_name,
                        "model": task.model_name,
                        "strategy": strategy,
                        "k_pct": float(k_pct),
                        "fold": fold_num,
                        "repeat_seed": task.repeat_seed,
                        "selected_features": "|".join(feature_names[selected_idx].tolist()),
                    }
                )
    return results, selections


def run_experiments(
    datasets: list[dict],
    run_config,
    existing_raw: pd.DataFrame,
    checkpoint_every_tasks: int = 1,
    checkpoint_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    done_keys = build_done_keys(existing_raw)
    result_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    tasks: list[tuple[ModelTask, dict]] = []
    for repeat_idx, repeat_seed in enumerate(run_config.repeat_seeds, start=1):
        for ds_idx, ds in enumerate(datasets, start=1):
            for model_idx, model_name in enumerate(MODELS, start=1):
                tasks.append(
                    (
                        ModelTask(
                            repeat_seed=repeat_seed,
                            repeat_idx=repeat_idx,
                            repeat_total=len(run_config.repeat_seeds),
                            dataset_idx=ds_idx,
                            dataset_total=len(datasets),
                            dataset_name=ds["name"],
                            model_idx=model_idx,
                            model_total=len(MODELS),
                            model_name=model_name,
                        ),
                        ds,
                    )
                )

    with ProcessPoolExecutor(max_workers=run_config.workers) as executor:
        future_map = {
            executor.submit(_run_model_task, task, ds, run_config, done_keys): (task, ds)
            for task, ds in tasks
        }
        progress = tqdm(total=len(future_map), desc="Model tasks")
        completed = 0
        for future in as_completed(future_map):
            task, _ = future_map[future]
            try:
                rows, sels = future.result()
                result_rows.extend(rows)
                selection_rows.extend(sels)
            except Exception as exc:  # noqa: BLE001
                logging.exception(
                    "Task failed repeat_seed=%d dataset=%s model=%s: %s",
                    task.repeat_seed,
                    task.dataset_name,
                    task.model_name,
                    exc,
                )
                raise
            finally:
                progress.update(1)
                completed += 1
                if checkpoint_callback is not None and completed % checkpoint_every_tasks == 0:
                    checkpoint_callback(result_rows, selection_rows)
        progress.close()

    if result_rows:
        new_df = pd.DataFrame(result_rows)
        new_df = new_df[RAW_COLUMNS]
        raw_df = pd.concat([existing_raw, new_df], ignore_index=True)
    else:
        raw_df = existing_raw.copy()

    selection_df = pd.DataFrame(selection_rows)
    return raw_df, selection_df
