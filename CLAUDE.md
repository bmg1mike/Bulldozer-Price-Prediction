# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ML project predicting bulldozer auction sale prices (Kaggle "Blue Book for Bulldozers" data) with scikit-learn. The real work lives in the Jupyter notebook `src/main.ipynb`; the `bulldozer_price_prediction_project` package is still the `uv init` stub (`main()` only prints a greeting).

## Environment & commands

Managed with `uv` (Python 3.12, see `.python-version`).

- Install/sync deps: `uv sync`
- Add a dependency: `uv add <pkg>` (updates `pyproject.toml` and `uv.lock`)
- Run the package entry point: `uv run bulldozer-price-prediction-project`
- The notebook runs on the `.venv` kernel. `ipykernel` is installed in `.venv` but isn't declared in `pyproject.toml`, so `uv sync` may remove it; use `uv add --dev ipykernel` if that happens.

No tests, linter, or formatter are configured.

## Data

- `src/data/TestAndValid.csv` (~120 MB, ~412k rows) holds the full training table, target column `SalePrice`. It is not gitignored, so don't commit it by accident.
- The notebook reads `data/TestAndValid.csv` by relative path, so its working directory must be `src/`.

## Notebook pipeline (`src/main.ipynb`)

The steps depend on each other in this order:

1. Load the CSV with `parse_dates=["saledate"]`, sort by `saledate`, and work on a copy (`dataset_temp`).
2. Feature engineering: split `saledate` into `SaleYear`/`SaleMonth`/`SaleDay`, then drop `saledate`.
3. Convert string columns to pandas `category`.
4. Missing values:
   - Numeric: add a `<col>_is_missing` indicator and fill with the median.
   - Categorical: add a `<col>_is_missing` indicator and replace values with `category.codes + 1`, so 0 means missing. `column_to_category_dict` keeps the code→label mapping.
5. Time-based split: `SaleYear == 2012` is the test/validation set and all earlier years are training. Keep the split temporal and don't swap in a random `train_test_split`.
6. Model: `RandomForestRegressor(n_jobs=-1, random_state=42)`.
7. Evaluation: `show_scores()` reports RMSLE (the competition metric, via `root_mean_squared_log_error`) plus R² and MAE.
