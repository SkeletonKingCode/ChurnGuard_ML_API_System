"""
Model predictor service handling artifact caching, feature transformation, and inference logic.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.serialize import load_model_artifacts
from src.features.feature_engineering import engineer_features
from src.api.schemas.customer import CustomerInput
from src.api.schemas.prediction import (
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchSummary,
)
from src.api.config import settings


class ModelPredictor:
    """Service class managing model artifacts and executing real-time predictions."""

    def __init__(self, version: str = settings.MODEL_VERSION):
        self.version = version
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.threshold = 0.50
        self.manifest = {}
        self.total_charges_median = settings.DEFAULT_TOTAL_CHARGES_MEDIAN
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load versioned model artifacts into memory with SHA-256 validation."""
        (
            self.model,
            self.preprocessor,
            self.feature_names,
            self.threshold,
            self.manifest,
        ) = load_model_artifacts(version=self.version, verify_checksum=True)

        # Attempt to load exact median TotalCharges from train set if available
        train_csv = settings.ROOT_DIR / "data/processed/train.csv"
        if train_csv.exists():
            try:
                train_df = pd.read_csv(train_csv)
                self.total_charges_median = float(train_df["TotalCharges"].median())
            except Exception:
                pass

    @property
    def is_loaded(self) -> bool:
        """Check if model and preprocessor are loaded in memory."""
        return self.model is not None and self.preprocessor is not None

    @staticmethod
    def classify_risk(probability: float, threshold: float) -> str:
        """Assign risk tier category based on predicted probability and decision threshold."""
        if probability >= 0.75:
            return "Critical"
        elif probability >= threshold:
            return "High"
        elif probability >= 0.30:
            return "Medium"
        else:
            return "Low"

    def predict_dataframe(self, df: pd.DataFrame) -> List[PredictionResponse]:
        """Perform feature engineering, transformation, and model scoring on a DataFrame."""
        if not self.is_loaded:
            raise RuntimeError("Model artifacts are not loaded.")

        customer_ids = df["CustomerID"].tolist() if "CustomerID" in df.columns else [None] * len(df)

        # Fill TotalCharges if None/NaN
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = df["TotalCharges"].fillna(self.total_charges_median)

        # Map binary string values ("Yes"/"No"/"Male"/"Female") to 1/0
        binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
        binary_cols = ["PaperlessBilling", "Dependents", "Partner", "PhoneService", "Gender"]
        df_copy = df.copy()
        for col in binary_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].map(lambda x: binary_map.get(x, x))

        # 1. Feature Engineering
        engineered_df = engineer_features(df_copy, total_charges_median=self.total_charges_median)

        if "Churn" in engineered_df.columns:
            engineered_df = engineered_df.drop(columns=["Churn"])

        # 2. Pipeline Preprocessing
        X_arr = self.preprocessor.transform(engineered_df)

        # 3. Model Inference
        probas = self.model.predict_proba(X_arr)[:, 1]
        preds = (probas >= self.threshold).astype(int)

        results = []
        for cid, prob, pred in zip(customer_ids, probas, preds):
            prob_val = float(prob)
            pred_val = int(pred)
            label = "Churn" if pred_val == 1 else "No Churn"
            risk = self.classify_risk(prob_val, self.threshold)

            results.append(
                PredictionResponse(
                    customer_id=str(cid) if cid is not None else "UNKNOWN",
                    churn_probability=round(prob_val, 4),
                    churn_prediction=pred_val,
                    churn_label=label,
                    decision_threshold=self.threshold,
                    risk_tier=risk,
                )
            )
        return results

    def predict_single(self, customer: CustomerInput) -> PredictionResponse:
        """Process a single customer prediction request."""
        df = pd.DataFrame([customer.model_dump()])
        return self.predict_dataframe(df)[0]

    def predict_batch(self, batch_request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Process a batch of customer prediction requests and compute summary metrics."""
        records_data = [rec.model_dump() for rec in batch_request.records]
        df = pd.DataFrame(records_data)
        predictions = self.predict_dataframe(df)

        total_records = len(predictions)
        churn_count = sum(1 for p in predictions if p.churn_prediction == 1)
        mean_prob = float(np.mean([p.churn_probability for p in predictions]))
        churn_rate = float(churn_count / total_records) if total_records > 0 else 0.0

        summary = BatchSummary(
            total_records=total_records,
            predicted_churn_count=churn_count,
            churn_rate=round(churn_rate, 4),
            mean_churn_probability=round(mean_prob, 4),
        )

        return BatchPredictionResponse(predictions=predictions, summary=summary)


# Global singleton predictor instance
_predictor_instance: Optional[ModelPredictor] = None


def get_predictor() -> ModelPredictor:
    """Retrieve or initialize the global predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ModelPredictor()
    return _predictor_instance
