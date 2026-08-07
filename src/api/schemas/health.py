"""
Health check and model info schemas.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check endpoint status response."""

    status: str = Field(..., description="Service status ('ok', 'degraded', 'error')", examples=["ok"])
    version: str = Field(..., description="API version string", examples=["1.0.0"])
    model_loaded: bool = Field(..., description="Whether model artifacts are loaded in memory", examples=[True])
    timestamp_utc: str = Field(..., description="Current ISO-8601 UTC timestamp")


class ModelInfoResponse(BaseModel):
    """Comprehensive model metadata response."""

    model_name: str = Field(..., description="Name of the deployed model")
    version: str = Field(..., description="Model release version (e.g. 'v1.0.0')")
    model_class: str = Field(..., description="Estimator class name (e.g. 'LogisticRegression')")
    optimal_threshold: float = Field(..., description="Decision threshold applied for classification")
    num_features: int = Field(..., description="Number of encoded features expected by the model")
    created_at_utc: Optional[str] = Field(None, description="Artifact creation timestamp")
    checksums_sha256: Dict[str, str] = Field(..., description="Cryptographic SHA-256 artifact hashes")
    environment: Dict[str, str] = Field(..., description="Environment dependency versions")
