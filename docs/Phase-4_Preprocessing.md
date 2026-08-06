# Phase 4: Data Preprocessing

**Project:** End-to-End Customer Churn Prediction (Telecom)
**Script:** `src/data/preprocess.py`

---

## Run
To run, install requirements and run the following command:
```bash
uv run src/data/preprocess.py 
```

---

## 1. Order of operations

Split happens **before** any fitting, to avoid leakage:

1. Load and merge the 3 raw sources on `CustomerID`.
2. Clean dtypes and encode simple binary columns (deterministic, no fitting involved).
3. Drop `CustomerID`.
4. Split into train (70%) / val (15%) / test (15%), stratified on `Churn`.
5. Fit the sklearn pipeline (imputers, scaler, one-hot encoder) on **train only**.
6. Apply the fitted pipeline to val and test.
7. Save arrays, CSVs, and the fitted pipeline.

Fitting the scaler/encoder on the full dataset before splitting would leak
information about val/test into training (e.g. the scaler's mean would be
computed partly from data the model should never see). This is why the
split happens first.

---

## 2. Missing values

`TotalCharges` had 11 blank values (found in Phase 2/3), all belonging to
customers with `Tenure = 0` — brand-new customers with no billing history.

- Coerced to numeric (`errors="coerce"` turns blanks into `NaN`).
- Imputed with the **median**, computed on the training split only.
- Median chosen over mean: `TotalCharges` is right-skewed (see EDA report),
  and the missing values are all near-zero-tenure customers, so a low,
  robust central value is more appropriate than a mean pulled up by
  long-tenure customers.

No other columns had missing values per the Phase 3 EDA report.

---

## 3. Encoding categorical features

Three different treatments, chosen by cardinality:

| Type | Columns | Treatment |
|---|---|---|
| Binary Yes/No or Male/Female | `PaperlessBilling`, `Dependents`, `Partner`, `PhoneService`, `Gender` | Mapped directly to 1/0 |
| Already numeric binary | `SeniorCitizen` | Passed through unchanged |
| Multi-category (3+ values) | `ContractType`, `PaymentMethod`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | One-hot encoded |
| Target | `Churn` | Mapped Yes/No -> 1/0 |

Two-value columns are mapped to a single 0/1 column instead of one-hot
encoded, since one-hot would just produce a redundant second column. This
keeps the feature space smaller, which matters given the 2GB memory
constraint from Phase 1.

`OneHotEncoder(handle_unknown="ignore")` is used so that if a category
appears in future production data that wasn't seen in training, the model
gets a row of zeros for that feature instead of the pipeline crashing.

---

## 4. Scaling numerical features

`Tenure`, `MonthlyCharges`, `TotalCharges` are scaled with `StandardScaler`
(zero mean, unit variance), fit on train only.

Binary and one-hot columns are **not** scaled — scaling a 0/1 column
doesn't help tree-based or linear models and just makes coefficients
harder to read.

---

## 5. Removed columns

- `CustomerID`: unique identifier, not a predictive feature. Dropped
  before the split (kept implicitly via the DataFrame index if it's ever
  needed to trace a prediction back to a customer).

---

## 6. Train / validation / test split

70% / 15% / 15%, **stratified on `Churn`** so all three splits keep the
~26.5% churn rate observed in EDA. A plain random split risks skewing the
minority class across splits, which would bias evaluation metrics like
recall (the project's primary KPI, Phase 1).

`random_state=42` for reproducibility.

Result on this dataset:

| Split | Rows | Churn rate |
|---|---:|---:|
| Train | 4,930 | 26.5% |
| Val | 1,056 | 26.5% |
| Test | 1,057 | 26.6% |

---

## 7. Pipeline artifact

The fitted `ColumnTransformer` is saved to `models/preprocessor.joblib`.
This is the same object that will be loaded at inference time (Phase 11)
to transform raw incoming requests — training-time and serving-time
preprocessing must use the identical fitted object, not a re-fit one, or
predictions will be wrong.

`models/feature_names.json` records the exact output column order, needed
for SHAP explainability (Phase 1 guardrail) since SHAP values need to be
mapped back to human-readable feature names.

---

## 8. Outputs

```
data/processed/
  train.csv, val.csv, test.csv       # unscaled, human-readable, for SHAP/debugging
  X_train.npy, X_val.npy, X_test.npy # model-ready feature arrays
  y_train.npy, y_val.npy, y_test.npy # target arrays

models/
  preprocessor.joblib                # fitted ColumnTransformer
  feature_names.json                 # 40 output feature names, in array order
```

Final feature count: **40** (3 numeric + 32 one-hot + 5 binary).

---
