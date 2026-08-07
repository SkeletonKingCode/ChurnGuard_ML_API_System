# Phase 7: Model Training

**Project:** End-to-End Customer Churn Prediction (Telecom)  
**Script:** `src/models/train.py`  
**Inputs:** `data/processed/{X,y}_{train,val}.npy` (Phase 5 engineered features)  

---

## Run
To run, install requirements and run the following command:
```bash
uv run src/models/train.py
```

---

## 1. Method

1. Load the Phase 5 engineered train/val arrays.
2. Instantiate the 3 Phase 6 candidates (Logistic Regression, Random
   Forest, XGBoost), each configured to handle the ~27% churn rate
   directly (`class_weight="balanced"` / `scale_pos_weight`).
3. Run 5-fold stratified cross-validation on the training split for each
   model, scoring recall, ROC-AUC, precision, and F1 — the four Phase 1
   KPIs.
4. Refit each model on the full training split and score once on the
   held-out validation split, as a check that CV results generalize.
5. Save fitted models, a comparison table, and a comparison chart.

Cross-validation is the primary comparison signal, not the single
validation score — 5 folds give a mean and standard deviation per metric,
which is far more reliable than one train/val split for choosing between
close candidates.

---

## 2. Results

5-fold CV on the training split (4,930 rows):

| Model | Recall | ROC-AUC | Precision | F1 |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.797 ± 0.020** | **0.845 ± 0.011** | 0.516 ± 0.019 | 0.626 ± 0.019 |
| Random Forest | 0.486 ± 0.021 | 0.833 ± 0.012 | **0.661 ± 0.015** | 0.560 ± 0.014 |
| XGBoost | 0.706 ± 0.031 | 0.833 ± 0.016 | 0.539 ± 0.026 | 0.611 ± 0.024 |

Held-out validation split (1,056 rows, single fit — sanity check only):

| Model | Recall | ROC-AUC | Precision | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.829 | 0.847 | 0.503 | 0.626 |
| Random Forest | 0.471 | 0.824 | 0.608 | 0.531 |
| XGBoost | 0.779 | 0.833 | 0.534 | 0.634 |

Chart: `docs/reports/figures/model_comparison.png` (CV means vs. the
Phase 1 KPI targets, shown as dashed reference lines).  
![Missing: Generate Chart by Running train.py](reports/figures/model_comparison.png)

Full numbers, including standard deviations: [docs/reports/model_comparison.csv](reports/model_comparison.csv)

---

## 3. Interpretation

**Logistic Regression is the strongest candidate on the primary KPI**,
clearing the recall target (0.797 vs. 0.75) and the ROC-AUC target
(0.845 vs. 0.85, effectively at target) at the default 0.5 threshold —
with no tuning yet applied. Validation results track the CV numbers
closely, so this isn't a fluke of one split.

**Random Forest underperforms on recall (0.486) despite
`class_weight="balanced"`.** This matches the risk flagged in Phase 6:
averaging votes across 300 trees pulls predicted probabilities toward
0.5, so fewer borderline churners cross the 0.5 decision threshold, even
though the model's ROC-AUC (0.833) shows it ranks customers by churn risk
almost as well as the other two. This is a threshold problem, not
necessarily a ranking problem — worth revisiting in Phase 8 by tuning the
decision threshold rather than discarding the model outright.

**XGBoost sits between the two** — recall (0.706) below target,
comparable ROC-AUC to Random Forest (0.833), and default hyperparameters
only (`n_estimators=300, max_depth=4, learning_rate=0.1`). Boosted trees
are known to be sensitive to exactly these hyperparameters, so this
result likely understates XGBoost's ceiling once tuned.

None of the three models yet clears **both** primary KPIs
simultaneously (Recall ≥ 0.75 AND ROC-AUC ≥ 0.85) — Logistic Regression
comes closest. This is expected: Phase 7 uses default/near-default
settings by design, so KPI compliance is properly done in Phase 8 (tuning), not in Phase 7.

---

## 4. Model selected for Phase 8 tuning

**Logistic Regression** carries forward as the primary tuning target: it
already meets the recall KPI untuned and is closest to the ROC-AUC KPI.
**XGBoost** should also be tuned as a secondary candidate — its
threshold-independent ranking quality (ROC-AUC) is close to Random
Forest's and its recall has the most obvious room to improve with
hyperparameter search (deeper trees, lower learning rate, more
estimators). Random Forest is deprioritized: its recall gap is the
largest, and it's driven by a structural property of bagged
probabilities, which tuning `n_estimators` or `max_depth` is less likely
to close than a threshold adjustment would.

---

## 5. Outputs

```
models/
  logistic_regression.joblib
  random_forest.joblib
  xgboost.joblib

docs/reports/
  model_comparison.csv        # CV mean/std + val score, all 4 metrics, all 3 models
  figures/model_comparison.png
```
