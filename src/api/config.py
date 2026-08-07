"""
Configuration settings for the Customer Churn Inference Service.
"""

from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application configuration and environment settings."""
    PROJECT_NAME: str = "Customer Churn Prediction API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    API_LATEST_PREFIX: str = "/api/latest"

    # Root paths
    ROOT_DIR: Path = Path(__file__).resolve().parents[2]
    MODELS_DIR: Path = ROOT_DIR / "models"
    MODEL_VERSION: str = "latest"  # 'latest' or specific version like 'v1.0.0'

    # Operational Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Default fallback median for TotalCharges if not provided in raw input
    DEFAULT_TOTAL_CHARGES_MEDIAN: float = 1397.475


settings = Settings()
