"""
Health check and model metadata route controllers.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status
from src.api.schemas.health import HealthResponse, ModelInfoResponse
from src.api.services.predictor import get_predictor
from src.api.config import settings

router = APIRouter(tags=["Health & System Metadata"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check probe",
    description="Returns current status of the service, API version, model load status, and UTC timestamp.",
)
async def health_check():
    """Service health and readiness probe."""
    try:
        predictor = get_predictor()
        model_loaded = predictor.is_loaded
        service_status = "ok" if model_loaded else "degraded"
    except Exception:
        model_loaded = False
        service_status = "error"

    return HealthResponse(
        status=service_status,
        version=settings.VERSION,
        model_loaded=model_loaded,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Model metadata information",
    description="Returns detailed metadata regarding the active deployed model version, hyperparameters, features, and SHA-256 hashes.",
)
async def model_info():
    """Retrieve metadata of the currently deployed model version."""
    predictor = get_predictor()
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model predictor is not initialized or loaded.",
        )

    manifest = predictor.manifest
    return ModelInfoResponse(
        model_name=manifest.get("model_name", "Customer Churn Logistic Regression"),
        version=manifest.get("version", settings.MODEL_VERSION),
        model_class=manifest.get("model_class", "LogisticRegression"),
        optimal_threshold=predictor.threshold,
        num_features=len(predictor.feature_names or []),
        created_at_utc=manifest.get("created_at_utc"),
        checksums_sha256=manifest.get("checksums_sha256", {}),
        environment=manifest.get("environment", {}),
    )
