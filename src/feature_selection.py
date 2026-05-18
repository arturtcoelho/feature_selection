from __future__ import annotations

import numpy as np
import shap
from shap import maskers
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.linear_model import Ridge

from src.models import make_model


def rank_mutual_info(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    scores = mutual_info_regression(X_train, y_train, random_state=seed)
    return np.argsort(scores)[::-1]


def rank_rfe(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    estimator = Ridge(alpha=1.0)
    selector = RFE(estimator=estimator, n_features_to_select=1, step=1)
    selector.fit(X_train, y_train)
    ranks = selector.ranking_
    return np.argsort(ranks)


def rank_tree_importance(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    model = ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return np.argsort(model.feature_importances_)[::-1]


def rank_shap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_name: str,
    sample_ratio: float,
    use_gpu: bool,
) -> np.ndarray:
    n_rows = X_train.shape[0]
    n_sample = max(50, int(round(n_rows * sample_ratio)))
    n_sample = min(n_rows, n_sample)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n_rows, size=n_sample, replace=False)
    X_sample = X_train[sample_idx]
    y_sample = y_train[sample_idx]

    model = make_model(model_name, seed=seed, use_gpu=use_gpu)
    model.fit(X_sample, y_sample)

    if model_name == "ridge":
        masker = maskers.Independent(X_sample)
        explainer = shap.LinearExplainer(model, masker=masker)
        shap_values = explainer.shap_values(X_sample)
    else:
        explainer = shap.TreeExplainer(model, data=X_sample, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_sample, check_additivity=False)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    return np.argsort(mean_abs)[::-1]


def select_top_k_indices(ranking: np.ndarray, n_features: int, k_pct: float) -> np.ndarray:
    k = max(1, int(round(n_features * k_pct)))
    return ranking[:k]


def strategy_rank(
    strategy: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_name: str,
    shap_sample_ratio: float,
    use_gpu: bool,
) -> np.ndarray:
    if strategy == "mi":
        return rank_mutual_info(X_train, y_train, seed)
    if strategy == "rfe":
        return rank_rfe(X_train, y_train, seed)
    if strategy == "tree":
        return rank_tree_importance(X_train, y_train, seed)
    if strategy == "shap":
        return rank_shap(X_train, y_train, seed, model_name, shap_sample_ratio, use_gpu)
    raise ValueError(f"Unknown strategy: {strategy}")
