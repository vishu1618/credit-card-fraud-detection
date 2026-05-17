"""
health.py
Health and model info endpoints.

GET /health      — liveness check (used by load balancers, Docker healthcheck)
GET /model-info  — model metadata and training metrics
"""

from fastapi import APIRouter, Depends
from app.dependencies import get_model_service
from app.schemas.response import HealthResponse, ModelInfoResponse
from app.services.model_service import ModelService
from app.config import APP_VERSION

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check(service: ModelService = Depends(get_model_service)):
    """
    Returns API status and whether the model is loaded.
    Always returns 200 if the server is up — model_loaded tells you if inference is ready.
    """
    return HealthResponse(
        status="healthy" if service._loaded else "degraded",
        model_loaded=service._loaded,
        version=APP_VERSION,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(service: ModelService = Depends(get_model_service)):
    """
    Returns model metadata: threshold, feature count, and training metrics
    from reports/model_report.json (if available).
    """
    return ModelInfoResponse(**service.get_info())