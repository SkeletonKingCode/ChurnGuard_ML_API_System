"""
Pytest global configuration and shared fixtures for unit, integration, validation, and performance tests.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.main import app


@pytest.fixture(scope="session")
def api_client():
    """FastAPI TestClient fixture for integration tests."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_single_payload():
    """Valid raw customer dictionary matching CustomerInput Pydantic schema."""
    return {
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


@pytest.fixture
def sample_batch_payload(sample_single_payload):
    """Valid batch payload dictionary containing multiple customer records."""
    second_customer = {
        "CustomerID": "5575-GNVDE",
        "Gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "Tenure": 34,
        "PhoneService": 1,
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "ContractType": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 56.95,
        "TotalCharges": 1889.50,
    }
    return {"records": [sample_single_payload, second_customer]}


@pytest.fixture
def sample_raw_df(sample_single_payload, sample_batch_payload):
    """Pandas DataFrame representing raw un-preprocessed customer records."""
    records = sample_batch_payload["records"]
    return pd.DataFrame(records)
