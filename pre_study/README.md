# Pre-study Data Report

This folder contains a reproducible data pre-study pipeline used before long-running experiments.

Pipeline:

1. Download datasets programmatically from OpenML/UCI mirrors
2. Preprocess and clean data with explicit rules
3. Save local CSV snapshots for reproducibility
4. Generate descriptive tables and correlation plots
5. Produce a consolidated markdown report

Run:

```bash
python3 pre_study/build_pre_study.py
```

Outputs:

- `pre_study/data/raw/*.csv`
- `pre_study/data/processed/*.csv`
- `pre_study/tables/*.csv`
- `pre_study/figures/*.png`
- `pre_study/report.md`
