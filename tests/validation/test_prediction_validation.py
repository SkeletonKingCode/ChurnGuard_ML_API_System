"""
Validation tests for prediction properties, probability bounds, risk tiers, and model determinism.
"""

import pandas as pd
from src.api.services.predictor import get_predictor


def test_model_determinism(sample_raw_df):
    """Test model produces 100% identical prediction probabilities for identical inputs."""
    predictor = get_predictor()

    preds1 = predictor.predict_dataframe(sample_raw_df)
    preds2 = predictor.predict_dataframe(sample_raw_df)

    for p1, p2 in zip(preds1, preds2):
        assert p1.churn_probability == p2.churn_probability
        assert p1.churn_prediction == p2.churn_prediction
        assert p1.risk_tier == p2.risk_tier


def test_prediction_threshold_consistency(sample_raw_df):
    """Test binary classification decision strictly respects the optimal threshold (0.49)."""
    predictor = get_predictor()
    predictions = predictor.predict_dataframe(sample_raw_df)

    threshold = predictor.threshold

    for pred in predictions:
        if pred.churn_probability >= threshold:
            assert pred.churn_prediction == 1
            assert pred.churn_label == "Churn"
        else:
            assert pred.churn_prediction == 0
            assert pred.churn_label == "No Churn"


def test_risk_tier_classification_bounds():
    """Test classify_risk static method maps probabilities to correct risk tiers."""
    predictor = get_predictor()
    th = 0.49

    assert predictor.classify_risk(0.15, th) == "Low"
    assert predictor.classify_risk(0.35, th) == "Medium"
    assert predictor.classify_risk(0.55, th) == "High"
    assert predictor.classify_risk(0.85, th) == "Critical"
