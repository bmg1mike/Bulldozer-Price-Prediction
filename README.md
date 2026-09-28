# Bulldozer Price Prediction

Predicting the auction sale price of used bulldozers from their characteristics and sale date, using a random forest regressor built with scikit-learn.

The data comes from the Kaggle [Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers) competition. Models are scored with the competition's metric, root mean squared log error (RMSLE).

## Project structure

```
src/
├── main.ipynb                          # Analysis, feature engineering, and modelling
├── data/
│   └── TestAndValid.csv                # Auction records with SalePrice (not included, see below)
└── bulldozer_price_prediction_project/ # Python package (placeholder)
```

## Getting started

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Add a Jupyter kernel to the environment if you don't already have one
uv add --dev ipykernel
```

Download the competition data from Kaggle and save the training file as `src/data/TestAndValid.csv`.

Open `src/main.ipynb` with the project's `.venv` as the kernel. The notebook loads the data with a relative path (`data/TestAndValid.csv`), so run it from the `src/` directory.

## Approach

1. **Load and sort.** Read the CSV with `saledate` parsed as a date and sort the records chronologically.
2. **Feature engineering.** Split `saledate` into `SaleYear`, `SaleMonth`, and `SaleDay`.
3. **Categorical encoding.** Turn text columns into pandas categories, then into integer codes. Code `0` means the value is missing.
4. **Missing values.** Fill missing numeric values with the column median. For every column that had missing values, add a `<column>_is_missing` indicator column.
5. **Train/validation split.** Split by time rather than at random. Sales from 2012 form the validation set and all earlier sales are the training set. This mirrors the real task of predicting future prices.
6. **Model.** Fit a `RandomForestRegressor`.
7. **Evaluation.** Report RMSLE, R², and mean absolute error on the 2012 validation set.

## Tech stack

- pandas and NumPy for data handling
- scikit-learn for modelling and evaluation
- Matplotlib and seaborn for visualisation
