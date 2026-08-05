# Data Dictionary — Customer Churn Prediction

Source: IBM Telco Customer Churn dataset. Split into 3 files to simulate
separate production systems. Join key: `CustomerID`.

## contracts.csv (contract management system)

| Field | Type | Description | Notes |
|---|---|---|---|
| CustomerID | string | Unique customer identifier | Join key |
| Tenure | int | Months the customer has stayed with the company | Range 0-72 |
| ContractType | categorical | Month-to-month / One year / Two year | |
| PaymentMethod | categorical | Electronic check / Mailed check / Bank transfer / Credit card | |
| PaperlessBilling | binary | Yes/No | |
| Churn | binary | Yes/No — target variable | Placed here since churn is a contract-termination event |

## demographics.csv (CRM system)

| Field | Type | Description | Notes |
|---|---|---|---|
| CustomerID | string | Unique customer identifier | Join key |
| Gender | categorical | Male/Female | |
| SeniorCitizen | binary | 0/1 | Only age signal in source data; no continuous Age field exists |
| Dependents | binary | Yes/No | |
| Partner | binary | Yes/No | |

## usage.csv (billing/usage system)

| Field | Type | Description | Notes |
|---|---|---|---|
| CustomerID | string | Unique customer identifier | Join key |
| MonthlyCharges | float | Current monthly charge | |
| TotalCharges | float | Total charged over tenure | 11 rows blank (all Tenure=0, new customers) — handle in Phase 4 |
| PhoneService | binary | Yes/No | |
| MultipleLines | categorical | Yes/No/No phone service | |
| InternetService | categorical | DSL/Fiber optic/No | |
| OnlineSecurity | categorical | Yes/No/No internet service | |
| OnlineBackup | categorical | Yes/No/No internet service | |
| DeviceProtection | categorical | Yes/No/No internet service | |
| TechSupport | categorical | Yes/No/No internet service | |
| StreamingTV | categorical | Yes/No/No internet service | |
| StreamingMovies | categorical | Yes/No/No internet service | |

## Target variable
`Churn` (contracts.csv) — Yes/No, encode as 1/0 in preprocessing.
Class balance: 1,869 churned / 5,174 retained (~26.5% positive rate). Imbalanced —
address per Section 7 guardrails (class_weight/SMOTE) in Phase 6-7, not now.

## Known data quality issues (found during Phase 2, deferred to Phase 4)
- `TotalCharges` has 11 blank values, all for customers with `Tenure = 0`
  (brand-new customers with no billing history yet). Loaded as string type
  because of this; needs numeric coercion + imputation in preprocessing.
- No independent verification of label accuracy — dataset is a static Kaggle
  snapshot, not a live extract, so there's no "as-of" audit trail to check.

## Design decisions made in this phase
1. **Age field**: handbook lists `Age` under demographics, but the source
   data only has binary `SeniorCitizen`. Fabricating an age value would
   introduce fake signal, so `Age` was dropped rather than synthesized.
2. **Churn placement**: not assigned to a file in the original spec. Placed
   in `contracts.csv` as the most defensible single owner.
3. **CustomerID casing**: standardized to `CustomerID` (source used
   `customerID`) for consistent joins across all three files.
