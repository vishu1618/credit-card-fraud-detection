"""
main.py
FastAPI application entrypoint.

Defines the app instance, lifespan (startup/shutdown), and registers all routers.
This file should stay clean — no business logic lives here.

Run with:
    uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import APP_TITLE, APP_DESCRIPTION, APP_VERSION
from app.services.model_service import ModelService
from app.routers import health, predict

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan: load model once at startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts.
    Loads all model artifacts into app.state before accepting requests.
    If artifacts are missing, startup fails with a clear error message.
    """
    logger.info("Starting up — loading model artifacts...")
    service = ModelService()

    try:
        service.load()
    except RuntimeError as e:
        logger.error(f"STARTUP FAILED: {e}")
        raise  # Crash the server intentionally — don't serve with no model

    app.state.model_service = service
    logger.info("Startup complete. API is ready to serve requests.")

    yield  # Server runs here

    logger.info("Shutting down.")


# ── App instance ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(predict.router)


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    """Redirect hint for the root URL."""
    return JSONResponse({
        "message": "Credit Card Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
    })