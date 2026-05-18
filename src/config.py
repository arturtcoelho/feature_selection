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
    corr_prune_threshold: float | None = None


def build_run_config(quick: bool) -> RunConfig:
    workers = max(1, int(os.environ.get("TFM_WORKERS", "3")))
    if quick:
        return RunConfig(
            quick=True,
            folds=QUICK_FOLDS,
            repeat_seeds=QUICK_REPEAT_SEEDS,
            k_levels=QUICK_K_LEVELS,
            workers=workers,
            corr_prune_threshold=None,
        )
    return RunConfig(
        quick=False,
        folds=FULL_FOLDS,
        repeat_seeds=FULL_REPEAT_SEEDS,
        k_levels=FULL_K_LEVELS,
        workers=workers,
        corr_prune_threshold=None,
    )


def paths(root: Path, run_id: str | None = None) -> dict[str, Path]:
    if run_id:
        run_root = root / "runs" / run_id
        outputs_dir = run_root / "outputs"
        figures_dir = run_root / "figures"
        logs_dir = run_root / "logs"
    else:
        run_root = root
        outputs_dir = root / "outputs"
        figures_dir = root / "figures"
        logs_dir = root / "runs"

    return {
        "root": run_root,
        "outputs": outputs_dir,
        "figures": figures_dir,
        "logs": logs_dir,
        "raw": outputs_dir / "results_raw.csv",
        "selections": outputs_dir / "selections_raw.csv",
        "importances": outputs_dir / "feature_importances_raw.csv",
        "summary": outputs_dir / "results_summary.csv",
        "stats": outputs_dir / "stats_tests.csv",
        "stability": outputs_dir / "stability_analysis.csv",
    }


DATASETS = ["superconductor", "communities_crime", "bike_sharing"]
MODELS = ["ridge", "extratrees", "xgboost"]
STRATEGIES = ["mi", "rfe", "tree", "shap"]
