"""
dependencies.py
FastAPI dependency injection for shared resources.

Routers use get_model_service() to access the ModelService instance
that was loaded at startup. This keeps routers thin and testable.
"""

from fastapi import Request
from app.services.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    """
    Dependency that retrieves the ModelService from app state.
    Injected into route handlers via FastAPI's Depends().

    The service is stored on app.state during lifespan startup,
    so it is always available and never reloaded per-request.
    """
    return request.app.state.model_service