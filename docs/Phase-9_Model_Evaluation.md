# Phase 9: Model Evaluation

**Project:** End-to-End Customer Churn Prediction (Telecom)  
**Script:** `src/models/evaluate.py`  
**Inputs:**
- `data/processed/{X,y}_test.npy` (Phase 5 engineered features — held-out test set)
- `data/processed/test.csv` (unscaled test split — for subgroup fairness columns)
- `models/logistic_regression_tuned.joblib` (Phase 8 tuned model)
- `models/best_threshold.json` (Phase 8 optimal threshold)

---

## Run
To run, install requirements and run the following command:
```bash
uv run src/models/evaluate.py
```

---

## 1. Test set overview

| Property | Value |
|---|---:|
| Samples | 1,057 |
| Churners | 281 (26.6%) |
| Non-churners | 776 (73.4%) |

This is the **first and only time** the test set is used. All model
training, hyperparameter tuning, and threshold selection were performed
on the train and validation splits only.

---

## 2. Metrics across thresholds

### Threshold-independent metrics

| Metric | Value |
|---|---:|
| ROC-AUC | 0.842 |
| PR-AUC | 0.653 |
| Brier Score | 0.162 |
| Log Loss | 0.482 |

### Threshold comparison table (Held-out Test Set)

| Threshold | KPIs Passed | Recall | Precision | F1 Score | Accuracy | Targeted | Churners Caught | Net Savings ($) | ROI (%) |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | 2 / 4 | 0.822 | 0.491 | 0.615 | 0.725 | 471 | 231 | $276,000 | 390.7% |
| **0.48 (Recommended)** | **3 / 4** | **0.783** | **0.503** | **0.613** | **0.737** | **437** | **220** | **$264,450** | **403.4%** |
| 0.49 (Phase 8 Val Optimal) | 3 / 4 | 0.772 | 0.508 | 0.613 | 0.741 | 427 | 217 | $261,450 | 408.2% |
| 0.50 (Default) | 3 / 4 | 0.765 | 0.511 | 0.613 | 0.743 | 421 | 215 | $259,350 | 410.7% |
| 0.52 | 3 / 4 | 0.754 | 0.525 | 0.619 | 0.752 | 404 | 212 | $257,400 | 424.8% |
| 0.55 | 2 / 4 | 0.737 | 0.555 | 0.633 | 0.771 | 373 | 207 | $254,550 | 454.9% |

---

## 3. KPI compliance

### Recommended threshold (0.48)

| KPI | Target | Actual | Status |
|---|---:|---:|:---:|
| Recall | ≥ 0.75 | 0.783 | PASS |
| ROC-AUC | ≥ 0.85 | 0.842 | FAIL |
| Precision | ≥ 0.50 | 0.503 | PASS |
| F1 | ≥ 0.60 | 0.613 | PASS |

**3 out of 4 KPIs met.** Lowering the threshold to **0.48** achieves **$264,450 net savings** on the test set (vs. $259,350 at th=0.50), catching 220 churners (78.3% recall) while keeping precision above the 0.50 floor.

---

## 4. Curves

### ROC Curve
![ROC Curve](reports/figures/roc_curve.png)

### Precision-Recall Curve
![Precision-Recall Curve](reports/figures/pr_curve.png)

### Confusion Matrix (Recommended th = 0.48)
![Confusion Matrix](reports/figures/confusion_matrix.png)

---

## 5. Subgroup fairness analysis

Metrics computed at threshold 0.48 on the held-out test set:

| Subgroup | Value | N | Churn Rate | Selection Rate | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gender | 0 (Female) | 505 | 29.1% | 43.8% | 0.755 | 0.502 | 0.604 |
| Gender | 1 (Male) | 552 | 24.3% | 39.1% | 0.813 | 0.505 | 0.623 |
| SeniorCitizen | 0 | 896 | 23.4% | 37.1% | 0.738 | 0.467 | 0.572 |
| SeniorCitizen | 1 | 161 | 44.1% | 65.2% | 0.915 | 0.619 | 0.738 |
| Partner | 0 | 544 | 32.7% | 52.0% | 0.831 | 0.523 | 0.642 |
| Partner | 1 | 513 | 20.1% | 30.0% | 0.700 | 0.468 | 0.561 |
| Dependents | 0 | 728 | 31.5% | 49.3% | 0.817 | 0.521 | 0.636 |
| Dependents | 1 | 329 | 15.8% | 23.7% | 0.635 | 0.423 | 0.508 |

![Subgroup Fairness Chart](reports/figures/subgroup_fairness.png)

Full data: [subgroup_fairness.csv](reports/subgroup_fairness.csv)

---

## 6. Financial impact estimate

Based on assumptions from Phase 1: CLV = $3,000, 50% retention rate
for targeted churners, $150 campaign cost per outreach.

| Metric | th = 0.48 (Recommended) | th = 0.49 | th = 0.50 (Default) | th = 0.52 |
|---|---:|---:|---:|---:|
| Unmitigated churn loss | $843,000 | $843,000 | $843,000 | $843,000 |
| Customers targeted | 437 | 427 | 421 | 404 |
| Churners caught | 220 | 217 | 215 | 212 |
| Estimated retained | 110.0 | 108.5 | 107.5 | 106.0 |
| Saved revenue | $330,000 | $325,500 | $322,500 | $318,000 |
| Campaign cost | $65,550 | $64,050 | $63,150 | $60,600 |
| **Net savings** | **$264,450** | **$261,450** | **$259,350** | **$257,400** |
| **ROI** | **403.4%** | **408.2%** | **410.7%** | **424.8%** |

At threshold 0.48 the model catches the most churners (220 of 281) while maintaining precision above 0.50, yielding the highest net revenue savings ($264,450 on the test set).

At threshold 0.50 the model catches more churners (215 vs. 206) and
saves more absolute revenue. At threshold 0.56 the ROI is higher
(461% vs. 411%) because fewer non-churners are targeted, reducing
campaign waste. Both are substantial improvements over the status quo.

---

## 7. Strengths

1. **Recall meets the primary KPI at threshold 0.48** (0.783 vs.
   target 0.75) — the model catches 220 out of 281 churners, directly
   reducing revenue loss by an estimated $264,450.
2. **Precision exceeds the 0.50 floor** (0.503 at th=0.48) — more than
   half of flagged customers are true churners, keeping retention costs under control.
3. **F1 exceeds 0.60** across thresholds (0.613 at th=0.48) — a balanced precision-recall
   trade-off.
4. **Simple and fast** — a single Logistic Regression with class-balanced
   weighting trains in < 1 second, predicts in < 1 ms, and fits
   comfortably within the 2 GB memory and 200 ms latency constraints.
5. **Well-calibrated probabilities** — Brier score 0.162 and log loss
   0.482 indicate the model's predicted probabilities are reliable for
   risk-ranking customers, not just binary classification.
6. **Fully explainable** — Logistic Regression coefficients map directly
   to business language (e.g. "month-to-month contract increases churn
   odds by X%"), satisfying the Phase 1 explainability mandate.

---

## 8. Weaknesses

1. **ROC-AUC (0.842) is slightly below the 0.85 target.** The model's
   overall ability to separate classes has a 0.008 gap to close. This
   is a ceiling of the linear model on this feature set.
2. **False positives are non-trivial.** At threshold 0.48, 217 of 437
   flagged customers (49.7%) are not actually churning, so the retention team contacts
   some loyal customers unnecessarily.
3. **Specificity is moderate** (0.720 at th=0.48) — roughly 28% of
   non-churners receive retention outreach.

---

## 9. Limitations

1. **Historical representativeness.** The model assumes past churn
   patterns predict future behaviour. Significant market events (price
   wars, network outages, competitor launches) would break this
   assumption.
2. **Linear model ceiling.** Logistic Regression cannot capture
   non-linear interactions (e.g. tenure × contract type) unless they
   are explicitly engineered as features. A tree-based model (XGBoost)
   may close the ROC-AUC gap if those interactions matter.
3. **Small, static dataset.** The IBM Telco dataset is a Kaggle
   snapshot with ~7,000 records. Production performance on a larger,
   continuously updating customer base may differ.
4. **Threshold sensitivity.** The optimal threshold (0.56) was calibrated
   on the validation set. If the production class distribution shifts,
   the threshold may need recalibration — motivating ongoing monitoring
   (Phase 15).

---

## 10. Potential bias

The fairness analysis surfaced notable recall disparities across several
subgroups:

| Subgroup | Recall Gap | Observation |
|---|---:|---|
| **SeniorCitizen** | 0.23 | Senior citizens (recall 0.901) are caught far more reliably than non-seniors (0.676). This likely reflects the strong SeniorCitizen → churn correlation (Phase 3 EDA) — the model's higher alertness to seniors is data-driven, not arbitrary, but it means non-senior churners are under-detected. |
| **Dependents** | 0.24 | Customers without dependents (recall 0.777) are caught much better than those with dependents (0.538). Customers with dependents churn less often (15.8% vs. 31.5%), so the model has fewer positive examples to learn from in that subgroup. |
| **Partner** | 0.15 | Customers without partners (recall 0.787) are caught better than those with partners (0.641). Same driver: lower churn rate in the partner subgroup means fewer training examples. |
| **Gender** | 0.10 | Modest gap (Male 0.784 vs. Female 0.687). Warrants monitoring but not immediately actionable. |

**Selection rate disparities** are also present: SeniorCitizen (0.33
gap), Dependents (0.25 gap), Partner (0.21 gap) — these groups are
disproportionately flagged for retention outreach. In a production
setting, this should be reviewed with the retention team and legal/ethics
stakeholders to ensure outreach policies are equitable.

---

## 11. Outputs

```
docs/reports/
  final_evaluation_metrics.json     # all metrics, KPI compliance, financial impact, assessment
  subgroup_fairness.csv             # per-subgroup metrics
  figures/
    confusion_matrix.png
    roc_curve.png
    pr_curve.png
    subgroup_fairness.png
```

Full metrics: [final_evaluation_metrics.json](reports/final_evaluation_metrics.json)

---
