"""
Phase 7 — Model Training for the Customer Churn dataset.

Trains the 3 candidate algorithms selected in Phase 6 (see
docs/Phase-6_Model_Selection.md) on the Phase 5 engineered feature set:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost

Each model is evaluated with 5-fold stratified cross-validation on the
training split (recall, ROC-AUC, precision, F1 — the Phase 1 KPIs), then
refit on the full training split and scored once on the held-out
validation split. Cross-validation, not the single val score, is what
Phase 8 tuning decisions should be based on — the val score here is a
sanity check that CV performance transfers, not the selection signal.

Outputs:
  - models/logistic_regression.joblib
  - models/random_forest.joblib
  - models/xgboost.joblib
  - docs/reports/model_comparison.csv (CV mean/std + val score per model)
  - docs/reports/figures/model_comparison.png

Usage:
    uv run src/models/train.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("docs/reports")
FIG_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
N_FOLDS = 5

# Phase 1 KPI targets, plotted as reference lines against CV results
KPI_TARGETS = {"recall": 0.75, "roc_auc": 0.85, "precision": 0.50, "f1": 0.60}

SCORING = ["recall", "roc_auc", "precision", "f1"]


def load_data() -> tuple:
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    return X_train, y_train, X_val, y_val


def build_models(y_train: np.ndarray) -> dict:
    """Instantiate the 3 Phase 6 candidates. Each uses class weighting /
    scale_pos_weight to account for the ~27% churn rate rather than
    resampling the data."""
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def cross_validate_models(models: dict, X_train: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
    """5-fold stratified CV for each candidate, scored on the Phase 1 KPI metrics."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        result = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            row[f"{metric}_mean"] = result[f"test_{metric}"].mean()
            row[f"{metric}_std"] = result[f"test_{metric}"].std()
        rows.append(row)
        print(f"{name}: recall {row['recall_mean']:.3f} +/- {row['recall_std']:.3f}  "
              f"roc_auc {row['roc_auc_mean']:.3f} +/- {row['roc_auc_std']:.3f}")
    return pd.DataFrame(rows).set_index("model")


def fit_and_score_on_val(models: dict, X_train, y_train, X_val, y_val) -> pd.DataFrame:
    """Refit each model on the full training split and score once on val,
    as a sanity check that CV results generalize."""
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        rows.append({
            "model": name,
            "val_recall": recall_score(y_val, y_pred),
            "val_roc_auc": roc_auc_score(y_val, y_proba),
            "val_precision": precision_score(y_val, y_pred),
            "val_f1": f1_score(y_val, y_pred),
        })
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    return pd.DataFrame(rows).set_index("model")


def plot_comparison(comparison: pd.DataFrame) -> None:
    metrics = ["recall_mean", "roc_auc_mean", "precision_mean", "f1_mean"]
    labels = ["Recall", "ROC-AUC", "Precision", "F1"]
    targets = [KPI_TARGETS["recall"], KPI_TARGETS["roc_auc"], KPI_TARGETS["precision"], KPI_TARGETS["f1"]]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, metric, label, target in zip(axes, metrics, labels, targets):
        ax.bar(comparison.index, comparison[metric], color="steelblue")
        ax.axhline(target, color="firebrick", linestyle="--", linewidth=1, label=f"KPI target ({target})")
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
    fig.suptitle(f"{N_FOLDS}-fold CV performance vs. Phase 1 KPI targets")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=120)
    plt.close(fig)


def run() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = load_data()
    print(f"train {X_train.shape}  val {X_val.shape}")

    models = build_models(y_train)

    print(f"\nrunning {N_FOLDS}-fold stratified cross-validation on train...")
    cv_results = cross_validate_models(models, X_train, y_train)

    print("\nrefitting on full train, scoring once on val...")
    val_results = fit_and_score_on_val(models, X_train, y_train, X_val, y_val)

    comparison = cv_results.join(val_results)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv")
    plot_comparison(cv_results)

    print("\nmodel comparison (CV mean, val score):")
    print(comparison[["recall_mean", "roc_auc_mean", "val_recall", "val_roc_auc"]].to_string())

    best_model = comparison["recall_mean"].idxmax()
    print(f"\nhighest CV recall: {best_model}")
    print(f"saved: models/logistic_regression.joblib, models/random_forest.joblib, models/xgboost.joblib")
    print(f"saved: {REPORTS_DIR / 'model_comparison.csv'}")
    print(f"saved: {FIG_DIR / 'model_comparison.png'}")


if __name__ == "__main__":
    run()
