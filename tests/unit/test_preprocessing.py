"""
Unit tests for data preprocessing module (src/data/preprocess.py).
"""

import pandas as pd
import numpy as np
from src.data.preprocess import clean, build_pipeline, get_feature_names


def test_clean_binary_mapping():
    """Test clean() function maps binary string columns to 1/0 deterministically."""
    raw_df = pd.DataFrame([
        {
            "CustomerID": "001",
            "Churn": "Yes",
            "PaperlessBilling": "Yes",
            "Dependents": "No",
            "Partner": "Yes",
            "PhoneService": "No",
            "Gender": "Female",
            "TotalCharges": "29.85",
        },
        {
            "CustomerID": "002",
            "Churn": "No",
            "PaperlessBilling": "No",
            "Dependents": "Yes",
            "Partner": "No",
            "PhoneService": "Yes",
            "Gender": "Male",
            "TotalCharges": "100.50",
        },
    ])

    cleaned_df = clean(raw_df)

    assert cleaned_df["Churn"].tolist() == [1, 0]
    assert cleaned_df["PaperlessBilling"].tolist() == [1, 0]
    assert cleaned_df["Dependents"].tolist() == [0, 1]
    assert cleaned_df["Partner"].tolist() == [1, 0]
    assert cleaned_df["Gender"].tolist() == [0, 1]  # Female=0, Male=1
    assert cleaned_df["TotalCharges"].dtype in [np.float64, np.float32, float]


def test_build_pipeline_transform():
    """Test build_pipeline() ColumnTransformer fits and transforms input data cleanly."""
    df = pd.DataFrame({
        "Tenure": [1, 12, 24],
        "MonthlyCharges": [20.0, 50.0, 80.0],
        "TotalCharges": [20.0, 600.0, 1920.0],
        "ContractType": ["Month-to-month", "One year", "Two year"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "PaperlessBilling": [1, 0, 1],
        "Dependents": [0, 1, 0],
        "Partner": [1, 0, 1],
        "PhoneService": [0, 1, 1],
        "Gender": [0, 1, 0],
        "SeniorCitizen": [0, 0, 1],
    })

    preprocessor = build_pipeline()
    X_arr = preprocessor.fit_transform(df)

    assert isinstance(X_arr, np.ndarray)
    assert X_arr.shape[0] == 3
    assert X_arr.shape[1] > 10

    names = get_feature_names(preprocessor)
    assert len(names) == X_arr.shape[1]
