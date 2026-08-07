# Phase 12: Testing

## Overview

The **Testing Framework** provides comprehensive automated quality control across all layers of the machine learning pipeline and API microservice. Built using `pytest` and `pytest-cov`, the test suite verifies data preprocessing pipeline transformations, feature engineering logic, model serialization integrity, FastAPI endpoint contracts, risk tier classifications, and latency SLAs.

---

## Test Directory Architecture (`tests/`)

The test suite is structured into modular domain-specific directories:

```
tests/
├── conftest.py                   # Session-wide pytest fixtures (TestClient, raw DataFrames, payloads)
├── unit/                         # Unit tests for individual data & serialization modules
│   ├── __init__.py
│   ├── test_preprocessing.py    # Clean binary mapping, ColumnTransformer pipeline & missing value tests
│   ├── test_feature_engineering.py # TotalServices count, AvgMonthlySpend, ChargeDiff & binning tests
│   └── test_serialization.py    # SHA-256 hash calculation, version loading & checksum tamper tests
├── integration/                  # Microservice API integration tests
│   ├── __init__.py
│   └── test_api_endpoints.py    # FastAPI endpoints (/health, /info, /predict, /predict/batch)
├── validation/                   # Model output invariants & probability bound tests
│   ├── __init__.py
│   └── test_prediction_validation.py # Threshold consistency, risk tier bounds & determinism tests
├── performance/                  # Latency & throughput benchmark tests
│   ├── __init__.py
│   └── test_performance.py       # Single prediction latency (<50ms) & batch throughput benchmark
└── test_api.py                   # Legacy/flat API integration test suite
```

---

## Test Suites Summary (27 Test Cases)

### 1. Unit Tests (`tests/unit/`)

* **Preprocessing (`test_preprocessing.py`)**:
  - `test_clean_binary_mapping`: Validates `clean()` converts `"Yes"`/`"No"` and `"Male"`/`"Female"` to `1`/`0`.
  - `test_build_pipeline_transform`: Validates `ColumnTransformer` handles missing values and produces expected numeric feature arrays.
* **Feature Engineering (`test_feature_engineering.py`)**:
  - `test_engineer_features_calculations`: Verifies `TotalServices`, `AvgMonthlySpend`, `ChargeDiff`, and `TenureBucket` feature math.
  - `test_engineer_features_tenure_zero_clip`: Verifies `Tenure=0` division-by-zero safety guard using `clip(lower=1)`.
* **Model Serialization (`test_serialization.py`)**:
  - `test_get_environment_info`: Verifies python, scikit-learn, joblib, numpy, pandas versions are captured.
  - `test_compute_sha256`: Verifies cryptographic SHA-256 checksum generation.
  - `test_load_model_artifacts_latest`: Verifies loading production release artifacts with SHA-256 hash checks.
  - `test_checksum_mismatch_detection`: Verifies `load_model_artifacts` raises `ValueError` when an artifact is tampered with or corrupted.

### 2. Integration Tests (`tests/integration/`)

* `test_health_check`: Validates `GET /health` probe status and model load state.
* `test_model_info`: Validates `GET /info` returns model metadata, threshold (`0.49`), and feature count (46).
* `test_predict_single_v1` & `test_predict_single_latest`: Validates `POST /api/v1/predict` and `/api/latest/predict` return valid churn probabilities, predictions, and risk tiers.
* `test_predict_batch_v1` & `test_predict_batch_latest`: Validates `POST /api/v1/predict/batch` and `/api/latest/predict/batch` process multiple records and compute batch metrics.
* `test_predict_invalid_schema_422`: Validates Pydantic schema validation returns HTTP 422 for invalid payloads.

### 3. Prediction & Data Validation Tests (`tests/validation/`)

* `test_model_determinism`: Confirms identical raw customer inputs produce 100% identical prediction probabilities.
* `test_prediction_threshold_consistency`: Confirms binary classification decisions strictly obey the optimal decision threshold (`0.49`).
* `test_risk_tier_classification_bounds`: Verifies probability range tier mapping:
  - `Low`: `< 0.30`
  - `Medium`: `0.30 - 0.49`
  - `High`: `0.49 - 0.75`
  - `Critical`: `>= 0.75`

### 4. Latency & Performance Benchmarks (`tests/performance/`)

* `test_single_prediction_service_latency`: Verifies single prediction inference completes in **< 50 ms**.
* `test_batch_prediction_service_latency`: Verifies 100-record batch prediction inference completes in **< 200 ms**.
* `test_api_single_endpoint_latency`: Verifies end-to-end HTTP API request latency completes in **< 100 ms**.

---

## Test Execution Guide

### Install Dev Dependencies

```bash
uv pip install pytest pytest-cov httpx
```

### Running the Full Test Suite with Coverage

```bash
uv run pytest --cov=src tests/
```

### Running Specific Test Suites

```bash
# Run unit tests only
uv run pytest tests/unit/

# Run API integration tests only
uv run pytest tests/integration/

# Run performance benchmarks only
uv run pytest tests/performance/ -s
```

---

## Automated Execution Results

```text
============================= test session starts ==============================
platform darwin -- Python 3.14.5, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/d-23-6840/Documents/Project 3/Code
plugins: cov-7.1.0, anyio-4.14.2
collected 27 items

tests/integration/test_api_endpoints.py .......                          [ 25%]
tests/performance/test_performance.py ...                                [ 37%]
tests/test_api.py ......                                                 [ 59%]
tests/unit/test_feature_engineering.py ..                                [ 66%]
tests/unit/test_preprocessing.py ..                                      [ 74%]
tests/unit/test_serialization.py ....                                    [ 88%]
tests/validation/test_prediction_validation.py ...                       [100%]

================================ tests coverage ================================
_______________ coverage: platform darwin, python 3.14.5-final-0 _______________

Name                                  Stmts   Miss  Cover
---------------------------------------------------------
src/api/__init__.py                       1      0   100%
src/api/config.py                        15      0   100%
src/api/main.py                          34      5    85%
src/api/routes/__init__.py                3      0   100%
src/api/routes/health.py                 23      4    83%
src/api/routes/predict.py                23      6    74%
src/api/schemas/__init__.py               4      0   100%
src/api/schemas/customer.py              24      0   100%
src/api/schemas/health.py                16      0   100%
src/api/schemas/prediction.py            21      0   100%
src/api/services/__init__.py              2      0   100%
src/api/services/predictor.py            88      5    94%
---------------------------------------------------------
TOTAL                                   569    196    66%
======================= 27 passed, 15 warnings in 0.73s ========================
```
