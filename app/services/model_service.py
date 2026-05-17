"""
model_service.py
Encapsulates all model logic: loading artifacts, preprocessing, and inference.

This class is instantiated ONCE at app startup and shared across all requests.
Routers call methods here — they never touch the model or scaler directly.
"""

import json
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd

from app.config import (
    MODEL_PATH,
    SCALER_PATH,
    THRESHOLD_PATH,
    REPORT_PATH,
    FEATURE_COLUMNS,
    SCALE_COLUMNS,
    APP_VERSION,
)

logger = logging.getLogger(__name__)


class ModelService:
    """
    Handles artifact loading and all inference logic.

    Lifecycle:
        service = ModelService()
        service.load()          # call once at startup
        result = service.predict_one(features_dict)
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold: float = 0.5
        self.report: dict = {}
        self._loaded: bool = False
        self._load_time: float = 0.0

    # ── Startup ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load all three model artifacts from disk.
        Called once during FastAPI lifespan startup.
        Raises RuntimeError with a clear message if any file is missing.
        """
        self._check_artifacts_exist()

        logger.info("Loading model artifacts...")
        t0 = time.time()

        self.model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from  : {MODEL_PATH}")

        self.scaler = joblib.load(SCALER_PATH)
        logger.info(f"Scaler loaded from : {SCALER_PATH}")

        with open(THRESHOLD_PATH) as f:
            self.threshold = json.load(f)["threshold"]
        logger.info(f"Threshold loaded   : {self.threshold} (from {THRESHOLD_PATH})")

        # Load report if available (optional — used by /model-info)
        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH) as f:
                self.report = json.load(f)
            logger.info(f"Report loaded from : {REPORT_PATH}")
        else:
            logger.warning(f"No report found at {REPORT_PATH} — /model-info will show partial data.")

        self._loaded = True
        self._load_time = time.time() - t0
        logger.info(f"All artifacts loaded in {self._load_time:.2f}s. API is ready.")

    def _check_artifacts_exist(self) -> None:
        """Raise a clear error if any required artifact file is missing."""
        missing = []
        for path in [MODEL_PATH, SCALER_PATH, THRESHOLD_PATH]:
            if not os.path.exists(path):
                missing.append(path)

        if missing:
            raise RuntimeError(
                f"Missing model artifacts: {missing}\n"
                "Run `python scripts/train_pipeline.py` first to generate them."
            )

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_one(self, features: dict) -> dict:
        """
        Run inference on a single transaction.

        Args:
            features: Dict with keys matching FEATURE_COLUMNS (V1-V28, Amount, Time)

        Returns:
            Dict with is_fraud, fraud_probability, threshold_used, model_version
        """
        t0 = time.time()
        df = self._prepare_features(features)
        proba = float(self.model.predict_proba(df)[0][1])
        is_fraud = proba >= self.threshold
        latency_ms = round((time.time() - t0) * 1000, 2)

        return {
            "is_fraud": bool(is_fraud),
            "fraud_probability": round(proba, 6),
            "threshold_used": self.threshold,
            "model_version": APP_VERSION,
            "latency_ms": latency_ms,
        }

    def predict_batch(self, transactions: list[dict]) -> list[dict]:
        """
        Run inference on a list of transactions.

        Args:
            transactions: List of feature dicts

        Returns:
            List of prediction dicts in the same order as input.
        """
        t0 = time.time()
        df = pd.DataFrame([self._extract_ordered_features(t) for t in transactions])
        df[SCALE_COLUMNS] = self.scaler.transform(df[SCALE_COLUMNS])
        probas = self.model.predict_proba(df)[:, 1]
        total_ms = round((time.time() - t0) * 1000, 2)

        results = []
        for i, proba in enumerate(probas):
            proba = float(proba)
            results.append({
                "index": i,
                "is_fraud": bool(proba >= self.threshold),
                "fraud_probability": round(proba, 6),
                "threshold_used": self.threshold,
                "model_version": APP_VERSION,
            })

        logger.info(f"Batch prediction: {len(transactions)} transactions in {total_ms}ms")
        return results

    def get_info(self) -> dict:
        """Return model metadata for the /model-info endpoint."""
        info = {
            "model_name": "XGBoost Fraud Detector",
            "model_version": APP_VERSION,
            "threshold": self.threshold,
            "features": len(FEATURE_COLUMNS),
            "report_available": bool(self.report),
        }

        # Attach key metrics from the training report if available
        if self.report:
            metrics = self.report.get("metrics", {})
            info["roc_auc"] = metrics.get("ROC_AUC")
            info["recall"] = metrics.get("Recall")
            info["f1"] = metrics.get("F1")
            info["precision"] = metrics.get("Precision")

        return info

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _prepare_features(self, features: dict) -> pd.DataFrame:
        """
        Convert a features dict to a model-ready DataFrame.
        Applies scaling to Amount and Time using the saved scaler.
        """
        ordered = self._extract_ordered_features(features)
        df = pd.DataFrame([ordered])
        df[SCALE_COLUMNS] = self.scaler.transform(df[SCALE_COLUMNS])
        return df

    def _extract_ordered_features(self, features: dict) -> dict:
        """
        Extract features in the exact column order the model was trained on.
        This prevents column ordering bugs silently corrupting predictions.
        """
        try:
            return {col: features[col] for col in FEATURE_COLUMNS}
        except KeyError as e:
            raise ValueError(f"Missing required feature: {e}")