"""
Phase 8 — Hyperparameter Tuning for the Customer Churn dataset.

Tunes the best-performing model from Phase 7 (Logistic Regression) using
both Grid Search and Randomised Search, then applies threshold
optimisation on the validation split.

Strategy:
  1. Grid Search  — exhaustive sweep of C and l1_ratio with 5-fold
     stratified CV scored on ROC-AUC (the threshold-independent metric
     that measures ranking quality; recall is then recovered via
     threshold tuning in step 3).
  2. Random Search — 50 iterations over the same parameter space but
     with a continuous log-uniform C distribution, to demonstrate the
     alternative strategy and potentially find values the grid missed.
  3. Compare the best estimator from each search by CV ROC-AUC.
  4. Threshold optimisation — sweep [0.10, 0.90] on the validation split
     to find the threshold that maximises F1 while keeping recall ≥ 0.75
     (the Phase 1 floor).  This is NOT fit on training data, so it
     doesn't violate the CV principle.

Outputs:
  - models/logistic_regression_tuned.joblib
  - models/best_threshold.json
  - docs/reports/hyperparameter_tuning.csv
  - docs/reports/figures/tuning_comparison.png
  - docs/reports/figures/threshold_sweep.png

Usage:
    uv run src/models/tune.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
)
from scipy.stats import loguniform

# ── paths ────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("docs/reports")
FIG_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
N_FOLDS = 5

# Phase 1 KPI targets (used for reference lines on charts)
KPI_TARGETS = {"recall": 0.75, "roc_auc": 0.85, "precision": 0.50, "f1": 0.60}


# ── data ─────────────────────────────────────────────────────────────────
def load_data() -> tuple:
    """Load Phase 5 engineered train/val arrays."""
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    return X_train, y_train, X_val, y_val


# ── search helpers ───────────────────────────────────────────────────────
def _build_param_grid() -> dict:
    """Discrete parameter grid for GridSearchCV.

    Uses ``l1_ratio`` (sklearn ≥ 1.8) instead of the deprecated
    ``penalty`` parameter.  ``l1_ratio=0`` → L2, ``l1_ratio=1`` → L1.
    ``liblinear`` solver is used for both since it natively supports
    L1 and L2 without convergence issues on small datasets.
    """
    return {
        "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "l1_ratio": [0.0, 1.0],
    }


def _build_param_distributions() -> dict:
    """Continuous distributions for RandomizedSearchCV."""
    return {
        "C": loguniform(1e-3, 1e1),
        "l1_ratio": [0.0, 1.0],
    }


def _base_estimator() -> LogisticRegression:
    """Base Logistic Regression estimator.

    Uses ``liblinear`` solver, which natively supports both L1 and L2
    regularisation and converges reliably on this dataset size.
    ``class_weight='balanced'`` handles the ~27 % churn imbalance.
    """
    return LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )


def run_grid_search(
    X_train: np.ndarray, y_train: np.ndarray,
) -> GridSearchCV:
    """Exhaustive grid search over C and l1_ratio, scored on ROC-AUC."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=_base_estimator(),
        param_grid=_build_param_grid(),
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    return gs


def run_random_search(
    X_train: np.ndarray, y_train: np.ndarray, n_iter: int = 50,
) -> RandomizedSearchCV:
    """Randomised search with continuous C distribution, scored on ROC-AUC."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(
        estimator=_base_estimator(),
        param_distributions=_build_param_distributions(),
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    rs.fit(X_train, y_train)
    return rs


# ── threshold optimisation ───────────────────────────────────────────────
# Phase 1 financial assumptions — used to pick the threshold that
# maximises net savings, not an abstract metric like F1.
CLV = 3000
RETENTION_RATE = 0.50
CAMPAIGN_COST_PER_HEAD = 150


def sweep_thresholds(
    y_true: np.ndarray, y_proba: np.ndarray,
    low: float = 0.10, high: float = 0.90, step: float = 0.01,
) -> pd.DataFrame:
    """Evaluate recall, precision, F1, and estimated net savings at every
    candidate threshold.  Net savings = revenue retained − campaign cost,
    which is the metric the business actually cares about."""
    rows = []
    for th in np.arange(low, high + step, step):
        y_pred = (y_proba >= th).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        targeted = tp + fp
        saved_revenue = tp * RETENTION_RATE * CLV
        campaign_cost = targeted * CAMPAIGN_COST_PER_HEAD
        net_savings = saved_revenue - campaign_cost
        rows.append({
            "threshold": round(th, 2),
            "recall": recall_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "net_savings": net_savings,
        })
    return pd.DataFrame(rows)


def pick_best_threshold(sweep_df: pd.DataFrame, min_recall: float = 0.75, min_precision: float = 0.50) -> float:
    """Select the threshold that maximises estimated net savings while
    keeping recall >= 0.75 AND precision >= 0.50 (Phase 1 KPI constraints).
    This ensures business financial optimization without violating precision
    or recall operational targets."""
    valid = sweep_df[(sweep_df["recall"] >= min_recall) & (sweep_df["precision"] >= min_precision)]
    if valid.empty:
        valid = sweep_df[sweep_df["recall"] >= min_recall]
    if valid.empty:
        print(f"  WARNING: no threshold keeps recall ≥ {min_recall}; "
              f"using threshold with max recall instead")
        return sweep_df.loc[sweep_df["recall"].idxmax(), "threshold"]
    return valid.loc[valid["net_savings"].idxmax(), "threshold"]


# ── scoring helpers ──────────────────────────────────────────────────────
def score_model(
    model, X: np.ndarray, y: np.ndarray, threshold: float = 0.50,
) -> dict:
    """Score a fitted model on a given split at a given threshold."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "recall": recall_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }


def _format_l1_ratio(val: float) -> str:
    """Human-readable label for l1_ratio."""
    if val == 0.0:
        return "L2"
    if val == 1.0:
        return "L1"
    return f"ElasticNet({val})"


# ── visualisation ────────────────────────────────────────────────────────
def plot_tuning_comparison(comparison: pd.DataFrame) -> None:
    """Grouped bar chart: before vs. after tuning, one cluster per metric."""
    metrics = ["recall", "roc_auc", "precision", "f1"]
    labels = ["Recall", "ROC-AUC", "Precision", "F1"]
    targets = [KPI_TARGETS[m] for m in metrics]

    x = np.arange(len(metrics))
    n_stages = len(comparison)
    width = 0.8 / n_stages
    colours = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (stage, row) in enumerate(comparison.iterrows()):
        vals = [row[m] for m in metrics]
        offset = (i - n_stages / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=stage, color=colours[i % len(colours)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    for j, target in enumerate(targets):
        ax.hlines(target, x[j] - 0.45, x[j] + 0.45, colors="firebrick",
                  linestyles="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Phase 8: Hyperparameter Tuning — Before vs. After")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tuning_comparison.png", dpi=150)
    plt.close(fig)


def plot_threshold_sweep(sweep_df: pd.DataFrame, best_th: float) -> None:
    """Line chart of recall, precision, F1, and net savings vs. threshold."""
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Left axis: classification metrics
    ax1.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", color="#4C72B0", linewidth=2)
    ax1.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color="#55A868", linewidth=2)
    ax1.plot(sweep_df["threshold"], sweep_df["f1"], label="F1", color="#C44E52", linewidth=2)

    ax1.axvline(best_th, color="grey", linestyle=":", linewidth=1.5,
               label=f"Optimal threshold ({best_th:.2f})")
    ax1.axhline(KPI_TARGETS["recall"], color="firebrick", linestyle="--",
               linewidth=1, alpha=0.6, label=f"Recall floor ({KPI_TARGETS['recall']})")

    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Score")
    ax1.set_xlim(0.10, 0.90)
    ax1.set_ylim(0, 1.05)

    # Right axis: net financial savings
    ax2 = ax1.twinx()
    ax2.plot(sweep_df["threshold"], sweep_df["net_savings"] / 1000,
             label="Net savings ($k)", color="#8172B2", linewidth=2, linestyle="-.")
    ax2.set_ylabel("Net savings ($k)")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    ax1.set_title("Threshold Sweep — Metrics & Financial Impact (Validation Set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_sweep.png", dpi=150)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────
def run() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = load_data()
    print(f"train {X_train.shape}  val {X_val.shape}")

    # ── 1. Grid Search ──────────────────────────────────────────────────
    print("\n=== Grid Search (5-fold stratified CV, scored on ROC-AUC) ===")
    gs = run_grid_search(X_train, y_train)
    gs_l1r = _format_l1_ratio(gs.best_params_["l1_ratio"])
    print(f"  best params : C={gs.best_params_['C']}, {gs_l1r}")
    print(f"  best CV ROC-AUC: {gs.best_score_:.4f}")

    # ── 2. Random Search ────────────────────────────────────────────────
    print("\n=== Random Search (50 iter, 5-fold stratified CV, scored on ROC-AUC) ===")
    rs = run_random_search(X_train, y_train, n_iter=50)
    rs_l1r = _format_l1_ratio(rs.best_params_["l1_ratio"])
    print(f"  best params : C={rs.best_params_['C']:.6f}, {rs_l1r}")
    print(f"  best CV ROC-AUC: {rs.best_score_:.4f}")

    # ── 3. Pick winner by CV ROC-AUC ────────────────────────────────────
    gs_val = score_model(gs.best_estimator_, X_val, y_val)
    rs_val = score_model(rs.best_estimator_, X_val, y_val)

    if gs.best_score_ >= rs.best_score_:
        winner_label, winner_model = "Grid Search", gs.best_estimator_
        winner_params = gs.best_params_
    else:
        winner_label, winner_model = "Random Search", rs.best_estimator_
        winner_params = rs.best_params_

    print(f"\n  Winner: {winner_label} — C={winner_params['C']}, "
          f"{_format_l1_ratio(winner_params['l1_ratio'])}")

    # ── 4. Threshold optimisation ───────────────────────────────────────
    print("\n=== Threshold optimisation on validation set ===")
    y_proba_val = winner_model.predict_proba(X_val)[:, 1]
    sweep_df = sweep_thresholds(y_val, y_proba_val)
    best_th = pick_best_threshold(sweep_df)
    print(f"  optimal threshold: {best_th:.2f}")

    tuned_val = score_model(winner_model, X_val, y_val, threshold=best_th)
    print(f"  recall@{best_th}: {tuned_val['recall']:.4f}  "
          f"f1@{best_th}: {tuned_val['f1']:.4f}")

    # ── 5. Before/after comparison table ────────────────────────────────
    # Load the Phase 7 untuned LR to get its val scores for comparison
    untuned_path = MODELS_DIR / "logistic_regression.joblib"
    if untuned_path.exists():
        untuned_model = joblib.load(untuned_path)
        untuned_val = score_model(untuned_model, X_val, y_val)
    else:
        # Fallback: train an untuned LR from scratch
        print("  NOTE: Phase 7 logistic_regression.joblib not found — "
              "training an untuned baseline for comparison")
        untuned_model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        untuned_model.fit(X_train, y_train)
        untuned_val = score_model(untuned_model, X_val, y_val)

    rows = []
    rows.append({
        "stage": "Phase 7 (Untuned LR, th=0.50)",
        **untuned_val,
        "C": 1.0, "l1_ratio": 0.0, "threshold": 0.50,
    })
    rows.append({
        "stage": "Grid Search (th=0.50)",
        **gs_val,
        "C": gs.best_params_["C"],
        "l1_ratio": gs.best_params_["l1_ratio"],
        "threshold": 0.50,
    })
    rows.append({
        "stage": "Random Search (th=0.50)",
        **rs_val,
        "C": rs.best_params_["C"],
        "l1_ratio": rs.best_params_["l1_ratio"],
        "threshold": 0.50,
    })
    rows.append({
        "stage": f"Tuned LR + Optimal Threshold ({best_th:.2f})",
        **tuned_val,
        "C": winner_params["C"],
        "l1_ratio": winner_params["l1_ratio"],
        "threshold": best_th,
    })

    comparison = pd.DataFrame(rows).set_index("stage")
    comparison.to_csv(REPORTS_DIR / "hyperparameter_tuning.csv")

    print("\n=== Comparison (validation set) ===")
    print(comparison[["recall", "roc_auc", "precision", "f1"]].to_string())

    # ── 6. Save tuned model & metadata ──────────────────────────────────
    joblib.dump(winner_model, MODELS_DIR / "logistic_regression_tuned.joblib")

    meta = {
        "winner_search": winner_label,
        "best_params": {
            "C": float(winner_params["C"]),
            "l1_ratio": float(winner_params["l1_ratio"]),
            "regularisation": _format_l1_ratio(winner_params["l1_ratio"]),
            "solver": "liblinear",
        },
        "best_cv_roc_auc": float(max(gs.best_score_, rs.best_score_)),
        "optimal_threshold": float(best_th),
        "grid_search_best_cv_roc_auc": float(gs.best_score_),
        "random_search_best_cv_roc_auc": float(rs.best_score_),
    }
    with open(MODELS_DIR / "best_threshold.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── 7. Figures ──────────────────────────────────────────────────────
    plot_tuning_comparison(comparison)
    plot_threshold_sweep(sweep_df, best_th)

    print(f"\nsaved: {MODELS_DIR / 'logistic_regression_tuned.joblib'}")
    print(f"saved: {MODELS_DIR / 'best_threshold.json'}")
    print(f"saved: {REPORTS_DIR / 'hyperparameter_tuning.csv'}")
    print(f"saved: {FIG_DIR / 'tuning_comparison.png'}")
    print(f"saved: {FIG_DIR / 'threshold_sweep.png'}")


if __name__ == "__main__":
    run()
