# Phase 10: Model Serialization & Versioning

## Overview

Model serialization and versioning transition the trained machine learning pipeline from experimental scripts to a production-ready asset. In this phase, we package the tuned **Logistic Regression model**, fitted **ColumnTransformer preprocessing pipeline**, **feature name schema**, and **optimal decision threshold** into immutable, versioned artifact bundles.

All serialized artifacts undergo cryptographic checksum validation (SHA-256) upon reloading to guarantee data integrity, prevent model corruption, and enforce reproducible inference in downstream environments (such as FastAPI services or batch processing pipelines).

---

## Key Deliverables

1. **Serialized Trained Estimator**: Saved in both `Joblib` (`model.joblib`) and native `Pickle` (`model.pkl`) formats.
2. **Serialized Preprocessing Pipeline**: Fitted `ColumnTransformer` (`preprocessor.joblib` & `preprocessor.pkl`) covering missing value imputation, one-hot encoding, and feature scaling.
3. **Artifact Versioning**: Structured versioned release directory (`models/v1.0.0/`) alongside an updated production mirror (`models/latest/`).
4. **Metadata & Manifest Schema**: Comprehensive `metadata.json` capturing model hyperparameters, Python/library dependency versions, and SHA-256 hashes.
5. **Reload & Inference API**: Python module ([src/models/serialize.py](file:///Users/d-23-6840/Documents/Project%203/Code/src/models/serialize.py)) providing safe reloading with integrity verification and raw input inference (`predict_raw`).

---

## Serialization Strategy: Joblib vs. Pickle

We support both `Joblib` and `Pickle` formats to provide maximum flexibility and compatibility:

| Feature | Joblib (`.joblib`) | Pickle (`.pkl`) |
| :--- | :--- | :--- |
| **Primary Use Case** | Optimized for scikit-learn estimators & large NumPy arrays | Standard Python object serialization |
| **Performance** | Fast disk I/O & memory mapping for numeric arrays | Standard Python byte stream |
| **Dependency** | `joblib` third-party library | Python standard library (`pickle`) |
| **Role in Project** | **Primary deployment format** | Secondary fallback format |

### Security Considerations

> [!WARNING]
> **Untrusted Deserialization Warning**
> Both `pickle` and `joblib` execute arbitrary code during object deserialization. Never load `.joblib` or `.pkl` files received from untrusted external sources. Always verify file SHA-256 hashes against `metadata.json` prior to deserialization.

---

## Directory Structure & Versioning Policy

Model artifacts are stored in semantic versioned subdirectories under `models/`. The latest stable model release is automatically mirrored to `models/latest/` for simple access by inference services.

```
models/
├── v1.0.0/
│   ├── model.joblib                 # Tuned Logistic Regression estimator (Joblib)
│   ├── model.pkl                    # Tuned Logistic Regression estimator (Pickle)
│   ├── preprocessor.joblib          # Fitted ColumnTransformer pipeline (Joblib)
│   ├── preprocessor.pkl             # Fitted ColumnTransformer pipeline (Pickle)
│   ├── feature_names.json           # Ordered feature column names (46 features)
│   ├── best_threshold.json          # Optimal decision threshold (0.49) & search metadata
│   └── metadata.json                # Release manifest, dependency versions & SHA-256 hashes
├── latest/                          # Production mirror of the current active release
│   ├── model.joblib
│   ├── model.pkl
│   ├── preprocessor.joblib
│   ├── preprocessor.pkl
│   ├── feature_names.json
│   ├── best_threshold.json
│   └── metadata.json
├── best_threshold.json
├── feature_names.json
└── logistic_regression_tuned.joblib
```

---

## Metadata & Cryptographic Manifest (`metadata.json`)

Each version release includes a `metadata.json` file detailing model specifications, training environment versions, and SHA-256 checksums for integrity verification:

```json
{
  "model_name": "Customer Churn Logistic Regression (Tuned)",
  "version": "v1.0.0",
  "created_at_utc": "2026-08-07T20:04:44.123456+00:00",
  "serialization_formats": ["joblib", "pickle"],
  "primary_format": "joblib",
  "model_class": "LogisticRegression",
  "model_params": {
    "C": 1.0,
    "class_weight": "balanced",
    "dual": false,
    "fit_intercept": true,
    "intercept_scaling": 1,
    "l1_ratio": 1.0,
    "max_iter": 2000,
    "multi_class": "deprecated",
    "n_jobs": null,
    "penalty": "l2",
    "random_state": 42,
    "solver": "liblinear",
    "tol": 0.0001,
    "verbose": 0,
    "warm_start": false
  },
  "optimal_threshold": 0.49,
  "num_features": 46,
  "environment": {
    "python_version": "3.14.0",
    "scikit_learn_version": "1.7.1",
    "joblib_version": "1.4.2",
    "numpy_version": "2.2.3",
    "pandas_version": "2.2.3"
  },
  "checksums_sha256": {
    "model.joblib": "...",
    "model.pkl": "...",
    "preprocessor.joblib": "...",
    "preprocessor.pkl": "...",
    "feature_names.json": "...",
    "best_threshold.json": "..."
  }
}
```

---

## Model Reloading & Inference Code Guide

### 1. Reloading Serialized Artifacts

```python
from src.models.serialize import load_model_artifacts

# Load the latest production artifacts with cryptographic checksum validation
model, preprocessor, feature_names, threshold, manifest = load_model_artifacts(
    version="latest",
    verify_checksum=True,
    use_format="joblib"
)

print(f"Loaded {manifest['model_class']} (Version {manifest['version']})")
print(f"Decision Threshold: {threshold}")
```

### 2. End-to-End Prediction on Raw Customer Input

```python
import pandas as pd
from src.models.serialize import predict_raw

# Raw customer records (un-preprocessed)
raw_customers = pd.DataFrame([
    {
        "CustomerID": "7590-VHVEG",
        "Gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure": 1,
        "PhoneService": 0,
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "ContractType": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
])

# Run feature engineering, transformation, and thresholded prediction
predictions = predict_raw(raw_customers, version="latest")

print(predictions[["CustomerID", "churn_probability", "churn_prediction"]])
```

**Output:**
```text
  CustomerID  churn_probability  churn_prediction
  7590-VHVEG           0.642318                 1
```

---

## Verification & Validation

The serialization pipeline was validated by executing [src/models/serialize.py](file:///Users/d-23-6840/Documents/Project%203/Code/src/models/serialize.py):

- **Artifact Creation**: `models/v1.0.0/` and `models/latest/` created with all 7 required files.
- **SHA-256 Verification**: Successfully verified hash integrity on reload.
- **Inference Verification**: End-to-end raw customer predictions were evaluated against sample test records, achieving identical probabilities to the test split evaluation in Phase 9.
