from __future__ import annotations

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


def make_model(
    name: str,
    seed: int,
    use_gpu: bool = True,
    xgb_jobs: int | None = None,
    extratrees_jobs: int | None = None,
):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1 if extratrees_jobs is None else max(1, int(extratrees_jobs)),
        )
    if name == "xgboost":
        params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "random_state": seed,
            "verbosity": 0,
            "n_jobs": -1 if xgb_jobs is None else max(1, int(xgb_jobs)),
            "tree_method": "hist",
        }
        if use_gpu:
            params["device"] = "cuda"
        return XGBRegressor(**params)
    raise ValueError(f"Unknown model: {name}")
