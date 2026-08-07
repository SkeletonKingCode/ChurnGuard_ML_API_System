"""
Unit tests for feature engineering module (src/features/feature_engineering.py).
"""

import pandas as pd
import numpy as np
from src.features.feature_engineering import engineer_features, DROP_WEAK_COLS


def test_engineer_features_calculations():
    """Test engineer_features computes TotalServices, AvgMonthlySpend, ChargeDiff, and TenureBucket."""
    raw = pd.DataFrame([{
        "Tenure": 10,
        "MonthlyCharges": 70.0,
        "TotalCharges": 700.0,
        "PhoneService": 1,
        "InternetService": "Fiber optic",
        "MultipleLines": "Yes",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "ContractType": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": 1,
        "Dependents": 0,
        "Partner": 1,
        "Gender": 0,
        "SeniorCitizen": 0,
    }])

    fe = engineer_features(raw, total_charges_median=1397.475)

    # 1. TotalServices count:
    # MultipleLines=Yes (1) + OnlineBackup=Yes (1) + StreamingTV=Yes (1) + PhoneService (1) + InternetService!=No (1) = 5
    assert fe["TotalServices"].iloc[0] == 5

    # 2. AvgMonthlySpend: 700 / 10 = 70.0
    assert fe["AvgMonthlySpend"].iloc[0] == 70.0

    # 3. ChargeDiff: 70.0 - 70.0 = 0.0
    assert fe["ChargeDiff"].iloc[0] == 0.0

    # 4. TenureBucket: 10 months falls in "0-12"
    assert fe["TenureBucket"].iloc[0] == "0-12"

    # 5. Weak columns dropped
    for col in DROP_WEAK_COLS:
        assert col not in fe.columns


def test_engineer_features_tenure_zero_clip():
    """Test engineer_features guards against division by zero when Tenure is 0."""
    raw = pd.DataFrame([{
        "Tenure": 0,
        "MonthlyCharges": 50.0,
        "TotalCharges": np.nan,  # Missing total charges
        "PhoneService": 1,
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "ContractType": "Month-to-month",
        "PaymentMethod": "Mailed check",
        "PaperlessBilling": 0,
        "Dependents": 0,
        "Partner": 0,
        "Gender": 1,
        "SeniorCitizen": 0,
    }])

    fe = engineer_features(raw, total_charges_median=100.0)

    # TotalCharges filled with median 100.0
    assert fe["TotalCharges"].iloc[0] == 100.0
    # AvgMonthlySpend: 100.0 / clip(0, lower=1) = 100.0
    assert fe["AvgMonthlySpend"].iloc[0] == 100.0
    # ChargeDiff: 50.0 - 100.0 = -50.0
    assert fe["ChargeDiff"].iloc[0] == -50.0
