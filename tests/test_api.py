"""
Automated unit & integration tests for the FastAPI Inference Microservice.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Test GET /health returns 200 OK and model_loaded True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "version" in data
    assert "timestamp_utc" in data
    assert data["model_loaded"] is True


def test_model_info_endpoint():
    """Test GET /info returns model metadata."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "version" in data
    assert "optimal_threshold" in data
    assert data["num_features"] == 46


def test_predict_single_endpoint():
    """Test POST /api/v1/predict with valid customer payload."""
    payload = {
        "CustomerID": "TEST-1234",
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
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "TEST-1234"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in [0, 1]
    assert data["churn_label"] in ["Churn", "No Churn"]
    assert data["risk_tier"] in ["Low", "Medium", "High", "Critical"]


def test_predict_batch_endpoint():
    """Test POST /api/v1/predict/batch with multiple customer records."""
    payload = {
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
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["summary"]["total_records"] == 2
    assert "churn_rate" in data["summary"]


def test_predict_invalid_payload():
    """Test POST /api/v1/predict with invalid schema field returns 422 Unprocessable Entity."""
    invalid_payload = {
        "CustomerID": "TEST-BAD",
        "Gender": "InvalidGender",  # Invalid enum value
        "MonthlyCharges": -50.0      # Invalid negative charge
    }
    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_latest_alias_endpoint():
    """Test POST /api/latest/predict alias works identically to /api/v1/predict."""
    payload = {
        "CustomerID": "LATEST-1234",
        "Gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "Tenure": 12,
        "PhoneService": 1,
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "ContractType": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 45.0,
        "TotalCharges": 540.0
    }
    response = client.post("/api/latest/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "LATEST-1234"
    assert "churn_probability" in data
