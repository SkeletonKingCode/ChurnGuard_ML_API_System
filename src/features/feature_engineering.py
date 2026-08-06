"""
Phase 5 — Feature Engineering for the Customer Churn dataset.

Starts from the Phase 4 unscaled splits (data/processed/{train,val,test}.csv),
adds domain-driven features, drops features shown to carry near-zero signal,
then re-fits the encode+scale pipeline (train only) on the final feature set.

Outputs:
  - data/processed/X_train.npy, X_val.npy, X_test.npy
  - data/processed/y_train.npy, y_val.npy, y_test.npy (unchanged from Phase 4, kept alongside for convenience)
  - models/preprocessor.joblib (fitted ColumnTransformer, includes engineered features)
  - models/feature_names.json
  - docs/reports/feature_importance.csv (RandomForest importance, full ranking)
  - docs/reports/figures/feature_importance.png

Usage:
    uv run src/features/feature_engineering.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIG_DIR = Path("docs/reports/figures")
IMPORTANCE_PATH = Path("docs/reports/feature_importance.csv")

RANDOM_STATE = 42

# Services checked for the TotalServices count. PhoneService and
# InternetService are handled separately since they aren't plain Yes/No.
ADDON_SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "12-24", "24-48", "48-72"]

# Dropped after the importance analysis below (see docs/Phase-5_Feature_Engineering.md):
# both scored near-zero RandomForest importance and are redundant with
# MultipleLines / InternetService respectively.
DROP_WEAK_COLS = ["PhoneService", "HasInternetService"]

NUMERIC_COLS = ["Tenure", "MonthlyCharges", "TotalCharges", "TotalServices", "AvgMonthlySpend", "ChargeDiff"]
CATEGORICAL_COLS = [
    "ContractType", "PaymentMethod", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "TenureBucket",
]
BINARY_COLS = ["PaperlessBilling", "Dependents", "Partner", "Gender", "SeniorCitizen"]

TARGET_COL = "Churn"


def engineer_features(df: pd.DataFrame, total_charges_median: float) -> pd.DataFrame:
    """Add domain-driven features. Deterministic row-wise transforms only,
    except the TotalCharges fill value, which must come from the training
    split to avoid leakage (same value the Phase 4 pipeline learned)."""
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].fillna(total_charges_median)

    # How many services (of 9 possible) the customer has subscribed to.
    # A customer using more services has more invested in the relationship.
    df["TotalServices"] = (
        (df[ADDON_SERVICE_COLS] == "Yes").sum(axis=1)
        + df["PhoneService"]
        + (df["InternetService"] != "No").astype(int)
    )

    df["HasInternetService"] = (df["InternetService"] != "No").astype(int)

    # Actual historical average monthly spend, vs. the customer's *current*
    # rate (MonthlyCharges). Tenure=0 customers have no history yet, so
    # their average is just their current rate (division guarded with clip).
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["Tenure"].clip(lower=1)

    # Positive ChargeDiff = current rate is higher than the customer's own
    # historical average -> recent price increase, a plausible churn trigger.
    df["ChargeDiff"] = df["MonthlyCharges"] - df["AvgMonthlySpend"]

    df["TenureBucket"] = pd.cut(df["Tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)

    df = df.drop(columns=DROP_WEAK_COLS)
    return df


def build_pipeline() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("scale", StandardScaler())])
    categorical_pipeline = Pipeline(steps=[
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ("bin", "passthrough", BINARY_COLS),
    ])


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    cat_names = list(preprocessor.named_transformers_["cat"]
                      .named_steps["encode"].get_feature_names_out(CATEGORICAL_COLS))
    return NUMERIC_COLS + cat_names + BINARY_COLS


def analyze_importance(X: np.ndarray, y: np.ndarray, feature_names: list) -> pd.Series:
    """Fit a quick RandomForest for feature importance ranking. This is a
    diagnostic model only, not the Phase 6+ candidate model."""
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)


def plot_importance(importance: pd.Series, top_n: int = 20) -> None:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top.index, top.values, color="steelblue")
    ax.set_xlabel("RandomForest importance")
    ax.set_title(f"Top {top_n} features by importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    total_charges_median = train["TotalCharges"].median()

    train_fe = engineer_features(train, total_charges_median)
    val_fe = engineer_features(val, total_charges_median)
    test_fe = engineer_features(test, total_charges_median)

    y_train, y_val, y_test = train.pop(TARGET_COL), val.pop(TARGET_COL), test.pop(TARGET_COL)

    preprocessor = build_pipeline()
    X_train_arr = preprocessor.fit_transform(train_fe)
    X_val_arr = preprocessor.transform(val_fe)
    X_test_arr = preprocessor.transform(test_fe)

    feature_names = get_feature_names(preprocessor)
    print(f"features after engineering + encoding: {len(feature_names)}")

    # Save model-ready arrays, overwriting the Phase 4 arrays that will not be used with the new engineered features.
    np.save(PROCESSED_DIR / "X_train.npy", X_train_arr)
    np.save(PROCESSED_DIR / "X_val.npy", X_val_arr)
    np.save(PROCESSED_DIR / "X_test.npy", X_test_arr)
    np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
    np.save(PROCESSED_DIR / "y_val.npy", y_val.to_numpy())
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy())

    # Save the fitted pipeline and feature name order for inference-time reuse, overwriting the Phase 4 versions.
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    importance = analyze_importance(X_train_arr, y_train.to_numpy(), feature_names)
    importance.to_csv(IMPORTANCE_PATH, header=["importance"], index_label="feature")
    plot_importance(importance)

    print("\ntop 10 features by importance:")
    print(importance.head(10).to_string())
    print(f"\nsaved: data/processed/*.npy")
    print(f"saved: models/preprocessor.joblib, models/feature_names.json")
    print(f"saved: {IMPORTANCE_PATH}")
    print(f"saved: {FIG_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    run()
