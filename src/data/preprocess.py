"""
Phase 4 — Data Preprocessing for the Customer Churn dataset.

Loads the 3 raw source files, joins them, splits into train/val/test
BEFORE fitting any transformer (avoids leakage), then fits a single
sklearn preprocessing pipeline on train only and applies it to all splits.

Outputs:
  - data/processed/train.csv, val.csv, test.csv (human-readable, unscaled)
  - data/processed/X_train.npy, X_val.npy, X_test.npy (model-ready arrays)
  - data/processed/y_train.npy, y_val.npy, y_test.npy
  - models/preprocessor.joblib (fitted ColumnTransformer)
  - models/feature_names.json (column names after encoding, in array order)

Usage:
    uv run src/data/preprocess.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # of the full dataset, taken from the remaining train split

# Columns dropped entirely — identifiers carry no predictive signal
DROP_COLS = ["CustomerID"]

# Yes/No (and Male/Female) columns mapped directly to 1/0 instead of
# one-hot encoding, since they only have 2 categories
BINARY_MAP_COLS = [
    "PaperlessBilling", "Dependents", "Partner", "PhoneService", "Gender",
]
BINARY_VALUE_MAP = {
    "Yes": 1, "No": 0, "Male": 1, "Female": 0,
}

# Already-numeric binary column, passed through as-is
PASSTHROUGH_COLS = ["SeniorCitizen"]

# True numeric columns — impute + scale
NUMERIC_COLS = ["Tenure", "MonthlyCharges", "TotalCharges"]

# Multi-category columns (3+ values) — one-hot encode
CATEGORICAL_COLS = [
    "ContractType", "PaymentMethod", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]

TARGET_COL = "Churn"


def load_joined() -> pd.DataFrame:
    """Load and merge the three raw source CSV files."""
    contracts = pd.read_csv(RAW_DIR / "contracts.csv")
    demographics = pd.read_csv(RAW_DIR / "demographics.csv")
    usage = pd.read_csv(RAW_DIR / "usage.csv")
    df = contracts.merge(demographics, on="CustomerID").merge(usage, on="CustomerID")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes and encode simple binary columns. No leakage risk here —
    these are deterministic, row-independent transforms, not fitted on data."""
    df = df.copy()

    # TotalCharges was loaded as string because of 11 blank values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Target: Yes/No -> 1/0
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    # Simple binary columns: Yes/No or Male/Female -> 1/0
    for col in BINARY_MAP_COLS:
        df[col] = df[col].map(BINARY_VALUE_MAP)

    return df


def build_pipeline() -> ColumnTransformer:
    """Build the preprocessing pipeline. Fit only on the training split."""
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ("bin", "passthrough", BINARY_MAP_COLS + PASSTHROUGH_COLS),
    ])
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Flat list of output column names in the order the transformer emits them."""
    num_names = NUMERIC_COLS
    cat_names = list(preprocessor.named_transformers_["cat"]
                      .named_steps["encode"].get_feature_names_out(CATEGORICAL_COLS))
    bin_names = BINARY_MAP_COLS + PASSTHROUGH_COLS
    return num_names + cat_names + bin_names


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_joined()
    df = clean(df)
    df = df.drop(columns=DROP_COLS)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Split first: 70% train / 15% val / 15% test, stratified on target
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
        stratify=y_temp, random_state=RANDOM_STATE,
    )

    print(f"train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    # Fit the pipeline on train only, then apply to val/test
    preprocessor = build_pipeline()
    X_train_arr = preprocessor.fit_transform(X_train)
    X_val_arr = preprocessor.transform(X_val)
    X_test_arr = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)
    print(f"features after encoding: {len(feature_names)}")

    # Save model-ready arrays
    np.save(PROCESSED_DIR / "X_train.npy", X_train_arr)
    np.save(PROCESSED_DIR / "X_val.npy", X_val_arr)
    np.save(PROCESSED_DIR / "X_test.npy", X_test_arr)
    np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
    np.save(PROCESSED_DIR / "y_val.npy", y_val.to_numpy())
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy())

    # Save human-readable, unscaled splits (useful for SHAP later, debugging)
    X_train.assign(Churn=y_train).to_csv(PROCESSED_DIR / "train.csv", index=False)
    X_val.assign(Churn=y_val).to_csv(PROCESSED_DIR / "val.csv", index=False)
    X_test.assign(Churn=y_test).to_csv(PROCESSED_DIR / "test.csv", index=False)

    # Save the fitted pipeline and feature name order for inference-time reuse
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"train churn rate: {y_train.mean():.3f}")
    print(f"val churn rate:   {y_val.mean():.3f}")
    print(f"test churn rate:  {y_test.mean():.3f}")
    print("saved: data/processed/*.npy, data/processed/*.csv")
    print("saved: models/preprocessor.joblib, models/feature_names.json")


if __name__ == "__main__":
    run()
