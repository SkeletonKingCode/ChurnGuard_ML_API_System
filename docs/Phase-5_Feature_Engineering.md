# Phase 5: Feature Engineering

**Project:** End-to-End Customer Churn Prediction (Telecom)
**Script:** `src/features/feature_engineering.py`
**Inputs:** `data/processed/{train,val,test}.csv` (Phase 4 unscaled splits)

---

## Run
To run, install requirements and run the following command:
```bash
uv run src/features/feature_engineering.py 
```

---

## 1. New features created

| Feature | Definition | Rationale |
|---|---|---|
| `TotalServices` | Count of subscribed services (0-9): phone, internet, and 7 add-ons | A customer using more services has more switching cost and more invested in the relationship — a simple "stickiness" signal that no single raw column captures on its own. |
| `HasInternetService` | 1 if `InternetService != "No"` | Tested, then dropped — see Section 3. |
| `AvgMonthlySpend` | `TotalCharges / max(Tenure, 1)` | The customer's actual historical average spend, as opposed to their current listed rate. |
| `ChargeDiff` | `MonthlyCharges - AvgMonthlySpend` | Positive = the customer is currently paying more than their own historical average, a plausible proxy for a recent price increase — a known churn trigger. |
| `TenureBucket` | `Tenure` binned into 0-12 / 12-24 / 24-48 / 48-72 months | EDA (Phase 3) showed a non-linear tenure effect concentrated in the first year; bucketing lets linear models pick that up directly instead of relying on a single continuous coefficient. |

All are deterministic functions of raw columns — no leakage risk on their
own. The one exception is `TotalCharges` imputation feeding into
`AvgMonthlySpend`/`ChargeDiff`: the fill value is the **training-set**
median (matching the Phase 4 decision), applied identically to val/test.

---

## 2. Feature selection method

A diagnostic RandomForest (300 trees, not the Phase 6 candidate model) was
fit on the full engineered + encoded training set to rank features by
importance. One-hot dummy importances were summed back up to their source
column to judge each raw feature fairly (a single dummy's importance
understates a multi-category column's real contribution).

Full ranking: `docs/reports/feature_importance.csv`
Chart: `docs/reports/figures/feature_importance.png`

---

## 3. Weak features removed

| Feature | Importance | Why dropped |
|---|---:|---|
| `PhoneService` | 0.0026 (lowest of all features) | 90% of customers have it (per Phase 3 EDA) — almost no variance, and `MultipleLines` already encodes "No phone service" as its own category, making `PhoneService` redundant. |
| `HasInternetService` | 0.0029 (2nd lowest) | Fully redundant with `InternetService`'s "No" category — the model already gets this signal for free from the one-hot encoding. |

Everything else scored well above these two, including features one might
expect to be weak (`Gender` ≈ 0.021, `SeniorCitizen` ≈ 0.015) — those were
kept since they're not clear outliers on the low end.

---

## 4. Multicollinearity — flagged, not removed

`TotalCharges` correlates strongly with `Tenure` (r = 0.83, since
`TotalCharges` ≈ `Tenure × MonthlyCharges`), and `AvgMonthlySpend` /
`ChargeDiff` are strongly anti-correlated by construction (r = -0.86).

These were **kept** rather than dropped:
- Tree-based models (the Phase 6 candidates: RandomForest, XGBoost/LightGBM)
  are not harmed by correlated features — they just split importance
  between them, which is what the ranking shows.
- Dropping now would be premature: whether this matters depends on which
  algorithm wins in Phase 6. If a linear/logistic model turns out
  competitive, revisit then (e.g. drop `TotalCharges` in favor of
  `Tenure` + `MonthlyCharges`, since the former is nearly a product of
  the latter two).

---

## 5. Feature importance — top signals

1. **`TotalCharges` / `Tenure` / `AvgMonthlySpend` / `MonthlyCharges`** dominate — consistent with the Phase 3 EDA finding that tenure and pricing are the strongest churn correlates.
2. **`ChargeDiff`** (the new price-increase proxy) ranks 5th overall — the single strongest engineered feature.
3. **`ContractType_Month-to-month`** is the top categorical feature — matches Phase 3's finding that month-to-month has by far the highest churn rate (42.7% vs. 2.8% for two-year contracts).
4. **`TotalServices`** and **`TenureBucket_0-12`** both crack the top 10 — the engineered features are pulling real weight, not just adding noise.

---

## 6. Final feature set

**46 features** after encoding (up from 40 in Phase 4: +6 from
`TotalServices`, `AvgMonthlySpend`, `ChargeDiff`, `TenureBucket` [4
dummies], -2 from removing `PhoneService` and `HasInternetService`, net
of the one-hot expansion).

---

## 7. Outputs

```
data/processed/
  X_train.npy, X_val.npy, X_test.npy   # final model-ready arrays
  y_train.npy, y_val.npy, y_test.npy   # targets (same values as Phase 4)

models/
  preprocessor.joblib     # fitted pipeline incl. engineered features
  feature_names.json      # 46 output feature names, in array order

docs/reports/
  feature_importance.csv     # full RandomForest importance ranking
  figures/feature_importance.png
```

---

