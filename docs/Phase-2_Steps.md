# Phase 2: Data Collection:  

## 1. Install Requirements

```bash
uv venv
uv pip install -r requirements.txt
```

## 2. Download the raw dataset

```bash
cd Code
curl -s -o data/source/Telco-Customer-Churn.csv "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
```

Verify:

```bash
head -3 data/source/Telco-Customer-Churn.csv
wc -l data/source/Telco-Customer-Churn.csv
```

Expected: 7,043 data rows, 21 columns.

## 3. Inspect for data quality issues

```bash
uv run src/data/data_exploration.py
```

Findings: no NaN-type nulls, but 11 blank strings in `TotalCharges`
(all `Tenure = 0`). Class split: 5,174 No / 1,869 Yes.

## 4. Add the split script

Run:  
```bash
uv run src/data/split_sources.py
```

This reads `data/source/Telco-Customer-Churn.csv` and writes:
- `data/raw/contracts.csv`
- `data/raw/demographics.csv`
- `data/raw/usage.csv`

## 5. Verify the split

```bash
for f in contracts demographics usage; do
  echo "== $f.csv =="
  head -3 data/raw/$f.csv
  echo
done
```

## 6. Remove the temporary combined source file. (Optional)

```bash
rm data/source/Telco-Customer-Churn.csv
```

Only the three silo files should remain in `data/raw/`.

## 7. See Data dictionary

[docs/Data_Dictionary.md](Data_Dictionary.md) documents every
field, its source file, type, and the data-quality/design notes below.

## Design decisions made during this phase

1. **Age**: handbook lists `Age` under demographics, but the source data
   only has a binary `SeniorCitizen` flag. Dropped rather than fabricated.
2. **Churn placement**: not assigned to a file in the original spec.
   Placed in `contracts.csv` (churn is a contract-termination event).
3. **CustomerID casing**: standardized `customerID` -> `CustomerID` across
   all three files for consistent joins.

## Known issues carried into Phase 4

- `TotalCharges` loaded as string; 11 blanks need numeric coercion +
  imputation.
- No live "as-of" audit trail — this is a static Kaggle snapshot.

## Final structure after Phase 2

```
Code/
├── data/
│   ├── raw/
│   │   ├── contracts.csv
│   │   ├── demographics.csv
│   │   └── usage.csv
│   ├── processed/          (empty, used from Phase 4 onward)
│   └── DATA_DICTIONARY.md
├── notebooks/               (empty, used from Phase 3 onward)
├── src/
│   ├── data/
│   │   └── split_sources.py
│   ├── features/            (empty, used from Phase 5 onward)
│   ├── models/               (empty, used from Phase 6 onward)
│   └── api/                   (empty, used from Phase 11 onward)
├── tests/                     (empty, used from Phase 12 onward)
├── docker/                    (empty, used from Phase 13 onward)
└── scripts/                   (empty, used from Phase 14 onward)
```
