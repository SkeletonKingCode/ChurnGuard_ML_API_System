"""
FastAPI Inference Microservice application entry point for Customer Churn Prediction.

Usage:
    uv run src/api/main.py
    # Or with uvicorn CLI:
    uv run uvicorn src.api.main:app --reload --port 8000
"""

import sys
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure ROOT_DIR is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.config import settings
from src.api.routes import health_router, predict_router
from src.api.services.predictor import get_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup loading and shutdown cleanup."""
    print("Initializing Customer Churn Inference Microservice...")
    try:
        predictor = get_predictor()
        print(f"✓ Model '{predictor.manifest.get('model_name')}' loaded successfully (version {predictor.version}).")
        print(f"✓ Decision threshold: {predictor.threshold}")
    except Exception as e:
        print(f"❌ Failed to load model artifacts on startup: {e}")

    yield

    print("Shutting down Customer Churn Inference Microservice...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Production-grade RESTful API service exposing the tuned Customer Churn Prediction "
        "classification model. Provides single prediction, batch prediction, health checks, "
        "and metadata inspection."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers under both /api/v1 (version pinned) and /api/latest (alias)
app.include_router(health_router, prefix="")
app.include_router(predict_router, prefix=settings.API_PREFIX)
app.include_router(predict_router, prefix=settings.API_LATEST_PREFIX)


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred.",
            "error_message": str(exc),
            "path": str(request.url.path),
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
