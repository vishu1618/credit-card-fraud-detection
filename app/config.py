"""
config.py
Central configuration for the FastAPI application.
All paths and settings are defined here — no magic strings scattered across the codebase.
"""

import os

# ── Artifact paths ─────────────────────────────────────────────────────────────
# These must match what train_pipeline.py wrote in Phase 1.
# Paths are relative to the project root (where you run uvicorn from).
MODELS_DIR = "models"
REPORTS_DIR = "reports"

MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.json")
REPORT_PATH = os.path.join(REPORTS_DIR, "model_report.json")

# ── API metadata ───────────────────────────────────────────────────────────────
APP_TITLE = "Credit Card Fraud Detection API"
APP_DESCRIPTION = (
    "REST API for real-time fraud detection using a trained XGBoost classifier. "
    "Built on the Kaggle ULB Credit Card Fraud dataset."
)
APP_VERSION = "1.0.0"

# ── Feature configuration ──────────────────────────────────────────────────────
# The model expects features in this exact order.
# V1-V28 are PCA-transformed. Amount and Time are scaled by the saved scaler.
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
SCALE_COLUMNS = ["Amount", "Time"]

# ── Limits ─────────────────────────────────────────────────────────────────────
BATCH_MAX_SIZE = 100