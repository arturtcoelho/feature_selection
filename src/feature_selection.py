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


def score_mutual_info(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    return mutual_info_regression(X_train, y_train, random_state=seed)


def rank_rfe(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    estimator = Ridge(alpha=1.0)
    selector = RFE(estimator=estimator, n_features_to_select=1, step=1)
    selector.fit(X_train, y_train)
    ranks = selector.ranking_
    return np.argsort(ranks)


def score_rfe(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    estimator = Ridge(alpha=1.0)
    selector = RFE(estimator=estimator, n_features_to_select=1, step=1)
    selector.fit(X_train, y_train)
    ranks = selector.ranking_.astype(float)
    return ranks.max() - ranks + 1.0


def rank_tree_importance(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    model = ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return np.argsort(model.feature_importances_)[::-1]


def score_tree_importance(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    model = ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return model.feature_importances_


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


def score_shap(
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
    return np.mean(np.abs(shap_values), axis=0)


def select_top_k_indices(ranking: np.ndarray, n_features: int, k_pct: float) -> np.ndarray:
    k = max(1, int(round(n_features * k_pct)))
    return ranking[:k]


def prune_ranking_by_correlation(
    ranking: np.ndarray,
    X_train: np.ndarray,
    threshold: float | None,
) -> np.ndarray:
    if threshold is None:
        return ranking
    if threshold <= 0 or threshold >= 1:
        raise ValueError("corr_prune_threshold must be in (0, 1)")

    corr = np.corrcoef(X_train, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=1.0, neginf=1.0)
    kept: list[int] = []
    for feat_idx in ranking.tolist():
        if not kept:
            kept.append(int(feat_idx))
            continue
        max_corr = float(np.max(corr[int(feat_idx), kept]))
        if max_corr < threshold:
            kept.append(int(feat_idx))
    if not kept:
        kept = [int(ranking[0])]
    return np.array(kept, dtype=int)


def strategy_rank_with_scores(
    strategy: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_name: str,
    shap_sample_ratio: float,
    use_gpu: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "mi":
        scores = score_mutual_info(X_train, y_train, seed)
        return np.argsort(scores)[::-1], scores
    if strategy == "rfe":
        scores = score_rfe(X_train, y_train, seed)
        return np.argsort(scores)[::-1], scores
    if strategy == "tree":
        scores = score_tree_importance(X_train, y_train, seed)
        return np.argsort(scores)[::-1], scores
    if strategy == "shap":
        scores = score_shap(X_train, y_train, seed, model_name, shap_sample_ratio, use_gpu)
        return np.argsort(scores)[::-1], scores
    raise ValueError(f"Unknown strategy: {strategy}")
