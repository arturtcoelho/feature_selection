from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


SEED = 42
FULL_REPEAT_SEEDS = [42, 123, 256, 512, 999]
FULL_FOLDS = 10
FULL_K_LEVELS = [0.25, 0.50, 0.75, 1.00]

QUICK_REPEAT_SEEDS = [42, 123]
QUICK_FOLDS = 2
QUICK_K_LEVELS = [0.50, 1.00]


@dataclass(frozen=True)
class RunConfig:
    quick: bool
    folds: int
    repeat_seeds: list[int]
    k_levels: list[float]
    alpha: float = 0.05
    shap_sample_ratio: float = 0.10
    workers: int = 3
    use_gpu: bool = True


def build_run_config(quick: bool) -> RunConfig:
    workers = max(1, int(os.environ.get("TFM_WORKERS", "3")))
    if quick:
        return RunConfig(
            quick=True,
            folds=QUICK_FOLDS,
            repeat_seeds=QUICK_REPEAT_SEEDS,
            k_levels=QUICK_K_LEVELS,
            workers=workers,
        )
    return RunConfig(
        quick=False,
        folds=FULL_FOLDS,
        repeat_seeds=FULL_REPEAT_SEEDS,
        k_levels=FULL_K_LEVELS,
        workers=workers,
    )


def paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "outputs": root / "outputs",
        "figures": root / "figures",
        "logs": root / "runs",
        "raw": root / "outputs" / "results_raw.csv",
        "selections": root / "outputs" / "selections_raw.csv",
        "summary": root / "outputs" / "results_summary.csv",
        "stats": root / "outputs" / "stats_tests.csv",
        "stability": root / "outputs" / "stability_analysis.csv",
    }


DATASETS = ["superconductor", "communities_crime", "bike_sharing"]
MODELS = ["ridge", "extratrees", "xgboost"]
STRATEGIES = ["mi", "rfe", "tree", "shap"]
