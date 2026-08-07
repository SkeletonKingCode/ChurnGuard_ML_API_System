"""
Inference prediction route controllers.
"""

from fastapi import APIRouter, HTTPException, status
from src.api.schemas.customer import CustomerInput
from src.api.schemas.prediction import (
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from src.api.services.predictor import get_predictor

router = APIRouter(tags=["Churn Inference"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict churn risk for a single customer",
    description="Accepts raw customer attributes, runs feature engineering and preprocessing, and returns thresholded churn probability and risk tier.",
)
async def predict_single_customer(customer: CustomerInput):
    """Predict churn probability and binary classification for one customer record."""
    predictor = get_predictor()
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable.",
        )

    try:
        return predictor.predict_single(customer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {str(e)}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict churn risk for a batch of customers",
    description="Accepts a list of customer records (up to 1000 per request), performs vectorised prediction, and returns individual predictions plus batch summary statistics.",
)
async def predict_customer_batch(batch_request: BatchPredictionRequest):
    """Batch inference endpoint for processing multiple customer records in parallel."""
    predictor = get_predictor()
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable.",
        )

    try:
        return predictor.predict_batch(batch_request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch inference error: {str(e)}",
        )
