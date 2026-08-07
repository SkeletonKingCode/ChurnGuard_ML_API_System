"""
Integration tests for FastAPI inference endpoints (GET /health, GET /info, POST /predict, POST /predict/batch).
"""


def test_health_check(api_client):
    """Test GET /health probe returns 200 OK and model_loaded=True."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert data["model_loaded"] is True
    assert "timestamp_utc" in data


def test_model_info(api_client):
    """Test GET /info endpoint returns release manifest metadata."""
    response = api_client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "version" in data
    assert "optimal_threshold" in data
    assert data["num_features"] == 46


def test_predict_single_v1(api_client, sample_single_payload):
    """Test POST /api/v1/predict single prediction endpoint."""
    response = api_client.post("/api/v1/predict", json=sample_single_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == sample_single_payload["CustomerID"]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in [0, 1]
    assert data["churn_label"] in ["Churn", "No Churn"]
    assert data["risk_tier"] in ["Low", "Medium", "High", "Critical"]


def test_predict_single_latest(api_client, sample_single_payload):
    """Test POST /api/latest/predict alias route works identically to v1."""
    response = api_client.post("/api/latest/predict", json=sample_single_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == sample_single_payload["CustomerID"]
    assert "churn_probability" in data


def test_predict_batch_v1(api_client, sample_batch_payload):
    """Test POST /api/v1/predict/batch vectorised batch prediction endpoint."""
    response = api_client.post("/api/v1/predict/batch", json=sample_batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["summary"]["total_records"] == 2
    assert 0.0 <= data["summary"]["churn_rate"] <= 1.0


def test_predict_batch_latest(api_client, sample_batch_payload):
    """Test POST /api/latest/predict/batch alias route works identically to v1."""
    response = api_client.post("/api/latest/predict/batch", json=sample_batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["summary"]["total_records"] == 2


def test_predict_invalid_schema_422(api_client):
    """Test POST /api/v1/predict with invalid schema returns HTTP 422 Unprocessable Entity."""
    invalid_payload = {
        "CustomerID": "BAD-001",
        "Gender": "Robot",  # Invalid enum value
        "MonthlyCharges": -100.0,  # Out of range negative charge
    }
    response = api_client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422
