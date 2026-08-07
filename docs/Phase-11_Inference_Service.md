# Phase 11: Inference Service

## Overview

The **Inference Service** exposes the Phase 10 versioned Logistic Regression model via a production-ready RESTful API built with **FastAPI**, **Pydantic v2**, and **Uvicorn**.

The service loads the serialized model artifacts (`models/latest/`) at startup with SHA-256 integrity validation, executes real-time feature engineering and column transformations on raw input payloads, applies the business-optimized decision threshold (`0.49`), and returns thresholded classification decisions alongside risk tier assessments.

---

## Directory Architecture (`src/api/`)

The API codebase follows an enterprise modular layout separating routes, schemas, services, and configuration:

```
src/api/
├── __init__.py                   # Package metadata
├── main.py                       # FastAPI entrypoint, lifespan startup handler, CORS & router registration
├── config.py                     # App settings, paths, default thresholds, environment configuration
├── schemas/                      # Pydantic v2 validation models
│   ├── __init__.py
│   ├── customer.py               # Raw Customer input attributes, range checks, and enums
│   ├── prediction.py             # Single & Batch prediction response schemas and summary metrics
│   └── health.py                 # Health status and Model info metadata schemas
├── services/                     # Business logic layer
│   ├── __init__.py
│   └── predictor.py              # ModelPredictor class handling artifact caching, preprocessor & inference
└── routes/                       # FastAPI APIRouters
    ├── __init__.py
    ├── health.py                 # GET /health & GET /info route controllers
    └── predict.py                # POST /api/v1/predict & POST /api/v1/predict/batch route controllers
```

---

## API Endpoints Reference

The service exposes version-pinned endpoints under `/api/v1` as well as dynamic production alias endpoints under `/api/latest`:

| Method | Endpoint | Description | Response Model |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health & readiness probe | `HealthResponse` |
| `GET` | `/info` | Deployed model release metadata & checksums | `ModelInfoResponse` |
| `POST` | `/api/v1/predict` *(or `/api/latest/predict`)* | Single customer churn risk prediction | `PredictionResponse` |
| `POST` | `/api/v1/predict/batch` *(or `/api/latest/predict/batch`)* | Vectorised batch churn risk prediction (max 1000) | `BatchPredictionResponse` |

---

## Endpoint Specifications & Sample Payloads

### 1. Health Check (`GET /health`)

Returns real-time status of the service and model artifact load state.

**Sample Response (`200 OK`):**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "timestamp_utc": "2026-08-07T20:24:28.123456+00:00"
}
```

---

### 2. Model Information (`GET /info`)

Exposes details from the Phase 10 artifact manifest (`metadata.json`).

**Sample Response (`200 OK`):**
```json
{
  "model_name": "Customer Churn Logistic Regression (Tuned)",
  "version": "v1.0.0",
  "model_class": "LogisticRegression",
  "optimal_threshold": 0.49,
  "num_features": 46,
  "created_at_utc": "2026-08-07T20:04:44.123456+00:00",
  "checksums_sha256": {
    "model.joblib": "...",
    "preprocessor.joblib": "..."
  },
  "environment": {
    "python_version": "3.14.0",
    "scikit_learn_version": "1.9.0"
  }
}
```

---

### 3. Single Prediction (`POST /api/v1/predict`)

Predicts churn probability and assigns a risk tier for one customer.

**Request Payload:**
```json
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
  "TotalCharges": 29.85
}
```

**Response (`200 OK`):**
```json
{
  "customer_id": "7590-VHVEG",
  "churn_probability": 0.6423,
  "churn_prediction": 1,
  "churn_label": "Churn",
  "decision_threshold": 0.49,
  "risk_tier": "High"
}
```

---

### 4. Batch Prediction (`POST /api/v1/predict/batch`)

Processes an array of customer records in parallel (up to 1,000 per request).

**Request Payload:**
```json
{
  "records": [
    {
      "CustomerID": "CUST-001",
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
      "TotalCharges": 29.85
    },
    {
      "CustomerID": "CUST-002",
      "Gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "No",
      "Dependents": "No",
      "Tenure": 45,
      "PhoneService": 1,
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "Yes",
      "OnlineBackup": "Yes",
      "DeviceProtection": "Yes",
      "TechSupport": "Yes",
      "StreamingTV": "Yes",
      "StreamingMovies": "Yes",
      "ContractType": "Two year",
      "PaperlessBilling": "No",
      "PaymentMethod": "Credit card (automatic)",
      "MonthlyCharges": 105.65,
      "TotalCharges": 4754.25
    }
  ]
}
```

**Response (`200 OK`):**
```json
{
  "predictions": [
    {
      "customer_id": "CUST-001",
      "churn_probability": 0.6423,
      "churn_prediction": 1,
      "churn_label": "Churn",
      "decision_threshold": 0.49,
      "risk_tier": "High"
    },
    {
      "customer_id": "CUST-002",
      "churn_probability": 0.0481,
      "churn_prediction": 0,
      "churn_label": "No Churn",
      "decision_threshold": 0.49,
      "risk_tier": "Low"
    }
  ],
  "summary": {
    "total_records": 2,
    "predicted_churn_count": 1,
    "churn_rate": 0.5,
    "mean_churn_probability": 0.3452
  }
}
```

---

## Client Usage Examples

### Python Client (`httpx`)

```python
import httpx

payload = {
    "CustomerID": "CUST-100",
    "Gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "Tenure": 12,
    "PhoneService": 1,
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "ContractType": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 1146.00
}

response = httpx.post("http://127.0.0.1:8000/api/v1/predict", json=payload)
print(response.json())
```

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "CustomerID": "CUST-100",
       "Gender": "Male",
       "SeniorCitizen": 0,
       "Partner": "No",
       "Dependents": "No",
       "Tenure": 12,
       "PhoneService": 1,
       "MultipleLines": "Yes",
       "InternetService": "Fiber optic",
       "OnlineSecurity": "No",
       "OnlineBackup": "No",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "Yes",
       "StreamingMovies": "Yes",
       "ContractType": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 95.50,
       "TotalCharges": 1146.00
     }'
```

---

## How to Run the Server

### Development Mode (with hot-reloading)

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

### Production Mode

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Interactive OpenAPI documentation is available at:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`
