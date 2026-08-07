# Phase 8: Hyperparameter Tuning

**Project:** End-to-End Customer Churn Prediction (Telecom)  
**Script:** `src/models/tune.py`  
**Inputs:** `data/processed/{X,y}_{train,val}.npy` (Phase 5 engineered features)  

---

## Run
To run, install requirements and run the following command:
```bash
uv run src/models/tune.py
```

---

## 1. Model selected for tuning

**Logistic Regression** — the Phase 7 winner on recall (0.797 CV,
0.829 val at default threshold 0.50). See
[Phase-7_Model_Training.md](Phase-7_Model_Training.md) for the full
comparison.

---

## 2. Tuning strategy

Three complementary techniques applied in sequence:

### Grid Search (exhaustive)

Swept every combination of:

| Hyperparameter | Values |
|---|---|
| `C` (inverse regularisation strength) | 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0 |
| `l1_ratio` (0 = L2, 1 = L1) | 0.0, 1.0 |

- **14 combinations × 5 folds = 70 fits.**
- Scored on **ROC-AUC** (the threshold-independent measure of ranking
  quality). Recall is then recovered via threshold optimisation — this
  avoids the trap of optimising recall in CV, which can reward
  degenerate over-regularised models that predict everything positive.
- Solver: `liblinear` (natively supports both L1 and L2, converges
  reliably on this dataset size).

### Random Search (stochastic)

Same parameter space, but with a continuous **log-uniform** distribution
for `C` (range [10⁻³, 10¹]), sampling 50 random configurations:

- **50 iterations × 5 folds = 250 fits.**
- Demonstrates the alternative to grid search: covers the continuous
  `C` axis more densely in regions the grid may have skipped.

### Threshold optimisation (post-hoc, on validation set)

After selecting the best estimator, the decision threshold was swept
from 0.10 to 0.90 on the held-out validation split:

- For each threshold, recall, precision, F1, and estimated net financial savings (retained CLV − outreach costs) were computed.
- The threshold maximising **net financial savings** while satisfying both **recall ≥ 0.75** and **precision ≥ 0.50** (Phase 1 floors) was selected.
- This is NOT fit on training data — it's a post-hoc calibration step
  that directly aligns business revenue goals with model deployment.

---

## 3. Results

### Grid Search vs. Random Search

| Search | Best C | Regularisation | Best CV ROC-AUC |
|---|---:|---|---:|
| **Grid Search** | **1.0** | **L1** | **0.8450** |
| Random Search | 1.260 | L1 | 0.8450 |

Both searches converged to essentially the same configuration (C ≈ 1,
L1 regularisation). Grid Search is selected as the winner because it
achieved the same CV ROC-AUC with a rounder, simpler C value.

### Before vs. after comparison (validation set)

| Stage | Recall | ROC-AUC | Precision | F1 | Net Savings |
|---|---:|---:|---:|---:|---:|
| Phase 7 (Untuned LR, th=0.50) | 0.829 | 0.847 | 0.504 | 0.627 | $314,250 |
| Grid Search (th=0.50) | 0.825 | 0.847 | 0.502 | 0.624 | $312,600 |
| Random Search (th=0.50) | 0.825 | 0.847 | 0.502 | 0.624 | $312,600 |
| **Tuned LR + Optimal Threshold 0.49** | **0.836** | **0.847** | **0.503** | **0.628** | **$317,850** |

Full numbers: [hyperparameter_tuning.csv](reports/hyperparameter_tuning.csv)

### Tuning comparison chart

![Tuning Comparison](reports/figures/tuning_comparison.png)

### Threshold sweep

![Threshold Sweep](reports/figures/threshold_sweep.png)

---

## 4. Interpretation

**The untuned Logistic Regression was already near-optimal for this
dataset.** Both Grid Search and Random Search independently converged
to C=1.0 (the sklearn default) with L1 regularisation, producing
essentially identical ROC-AUC. This is consistent with the Phase 6
observation that the churn signal is largely linear and additive — a
simple model with default regularisation already captures it well.

**Financial optimization drove threshold selection.** In churn prevention, missing a churner carries an asymmetric cost (~$1,500 net lost CLV assuming 50% intervention success) compared to the campaign outreach cost ($150). Optimizing for financial savings subject to KPI constraints selected **th = 0.49**, capturing more at-risk customers (Recall 0.836 on validation) and delivering superior net financial returns compared to higher thresholds.

**L1 regularisation was preferred over L2.** This is expected: L1
performs implicit feature selection by driving weak coefficients to
exactly zero, which is beneficial given the 46-feature space includes
several near-zero-importance features from Phase 5 (e.g. streaming
service dummies). L2 keeps all coefficients non-zero, adding marginal
noise without improving the ranking.

---

## 5. Outputs

```
models/
  logistic_regression_tuned.joblib   # final tuned model (C=1.0, L1, liblinear)
  best_threshold.json                # optimal threshold + search metadata

docs/reports/
  hyperparameter_tuning.csv          # full before/after comparison table
  figures/tuning_comparison.png
  figures/threshold_sweep.png
```

---
