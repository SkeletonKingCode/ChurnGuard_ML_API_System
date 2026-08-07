# Phase 6: Model Selection

**Project:** End-to-End Customer Churn Prediction (Telecom)
**Input:** `data/processed/{X,y}_{train,val,test}.npy` (Phase 5 engineered features)
**Candidates implemented in:** `src/models/train.py` (Phase 7)

---

## 1. Selection criteria

Candidates are judged against the Phase 1 constraints, not general
popularity:

| Constraint (Phase 1) | Implication for model choice |
|---|---|
| Recall ≥ 0.75 (primary KPI) | Need a model whose probability output separates classes well enough to hit high recall without collapsing precision. |
| Explainability mandate (SHAP) | Every candidate must support SHAP (all 3 below do — linear SHAP or tree SHAP). |
| Inference latency < 200ms | Rules out heavy ensembles/stacking; all 3 candidates predict a single row in well under a millisecond. |
| Memory footprint (2GB) | Rules out very large ensembles; 300-tree forests and boosted trees fit comfortably. |
| 30-minute training budget | All 3 train on 4,930 rows x 46 features in seconds, leaving headroom for Phase 8 tuning. |

---

## 2. Candidates

### Logistic Regression — baseline
A linear model provides the reference point every other candidate must
beat. If a more complex model can't outperform it on the primary KPI,
that's a legitimate finding, not a failure to report.

- Coefficients map directly to business language (e.g. "month-to-month
  contract increases churn odds by X%"), which supports the
  explainability mandate with zero extra tooling.
- `class_weight="balanced"` addresses the ~27% churn rate without
  resampling the training data.
- Weakness: assumes a linear relationship between features and log-odds
  of churn, so it can't capture interactions (e.g. tenure x contract
  type) unless engineered explicitly.

### Random Forest
A bagged ensemble of decision trees, included to test whether nonlinear
interactions the linear model can't see actually improve on it.

- Captures nonlinear relationships and interactions natively.
- Robust to the outliers and skew flagged in Phase 3 EDA (e.g.
  `TotalCharges`) since splits are based on ordering, not magnitude.
- `class_weight="balanced"` reweights the minority (churn) class during
  training.
- Weakness: forest probability outputs tend to cluster near 0.5 (each
  tree votes, averaging pulls scores toward the center), which can hurt
  recall at the default 0.5 decision threshold — worth checking directly
  in Phase 7 rather than assuming it will help.

### XGBoost
A boosted ensemble, included as the model most likely to top out
performance on structured/tabular data.

- Sequential boosting typically outperforms bagging on tabular churn-style
  problems once tuned (Phase 8).
- `scale_pos_weight` (ratio of negative to positive class counts) handles
  the class imbalance directly, in place of `class_weight`.
- Built-in L1/L2 regularization reduces the overfitting risk that plain
  gradient boosting carries.
- Weakness: least interpretable of the three out of the box, though tree
  SHAP recovers per-prediction feature contributions just as well as for
  Random Forest.

---

## 3. Bias-variance progression

The three candidates are not arbitrary — they trace a deliberate
progression:

```
Logistic Regression   ->   Random Forest        ->   XGBoost
(high bias,                (lower bias,              (low bias,
 low variance)               variance-reduced           variance controlled
                             via bagging)                via regularization)
```

If the simplest model wins, that says the churn signal is largely
linear and additive. If the boosted model wins by a wide margin, that
says there's real nonlinear/interaction signal worth the added
complexity. Either outcome is informative — see Phase 7 for which one
actually happened on this dataset.

---

## 4. Not selected

- **KNN / SVM:** don't scale well past a few thousand rows without extra
  tuning, and neither is a natural fit for the mixed one-hot + numeric
  feature space here.
- **Neural network (MLP):** the dataset (4,930 training rows, 46
  features) is too small to justify one; it would very likely overfit
  and adds latency/serialization overhead for no expected benefit.
- **Naive Bayes:** its independence assumption is a poor fit given the
  correlated features flagged in Phase 5 (e.g. `TotalCharges` vs.
  `Tenure`).

---

## 5. Next step (Phase 7)

Train all 3 candidates with 5-fold stratified cross-validation on the
training split, compare against the Phase 1 KPI targets, and carry the
best performer into Phase 8 hyperparameter tuning.
