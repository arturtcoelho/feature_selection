# Pre-study Report

This report documents source download, preprocessing, local dataset snapshots, and exploratory summaries.

## Pipeline

download from source -> preprocess -> save as csv -> report -> use on study

## Superconductor

- Source key: `superconductor`
- Raw shape: `21263 x 81`
- Processed shape: `21263 x 81`
- Target: `critical_temp`

### Saved data

- `data/raw/superconductor_raw.csv`
- `data/processed/superconductor_processed.csv`

### Tables

- `tables/superconductor_overview.csv`
- `tables/superconductor_missingness.csv`
- `tables/superconductor_feature_stats.csv`
- `tables/superconductor_target_stats.csv`
- `tables/superconductor_correlation_matrix.csv`
- `tables/superconductor_top_target_correlations.csv`

### Figures

- `figures/superconductor_correlation_heatmap.png`
- `figures/superconductor_target_distribution.png`

## Communities and Crime

- Source key: `communities_crime`
- Raw shape: `1994 x 122`
- Processed shape: `1993 x 100`
- Target: `ViolentCrimesPerPop`

### Saved data

- `data/raw/communities_crime_raw.csv`
- `data/processed/communities_crime_processed.csv`

### Tables

- `tables/communities_crime_overview.csv`
- `tables/communities_crime_missingness.csv`
- `tables/communities_crime_feature_stats.csv`
- `tables/communities_crime_target_stats.csv`
- `tables/communities_crime_correlation_matrix.csv`
- `tables/communities_crime_top_target_correlations.csv`

### Figures

- `figures/communities_crime_correlation_heatmap.png`
- `figures/communities_crime_target_distribution.png`

## Bike Sharing

- Source key: `bike_sharing`
- Raw shape: `17379 x 12`
- Processed shape: `17379 x 12`
- Target: `cnt`

### Saved data

- `data/raw/bike_sharing_raw.csv`
- `data/processed/bike_sharing_processed.csv`

### Tables

- `tables/bike_sharing_overview.csv`
- `tables/bike_sharing_missingness.csv`
- `tables/bike_sharing_feature_stats.csv`
- `tables/bike_sharing_target_stats.csv`
- `tables/bike_sharing_correlation_matrix.csv`
- `tables/bike_sharing_top_target_correlations.csv`

### Figures

- `figures/bike_sharing_correlation_heatmap.png`
- `figures/bike_sharing_target_distribution.png`
