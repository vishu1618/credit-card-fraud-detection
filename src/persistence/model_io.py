"""
model_io.py
Handles saving and loading of all model artifacts.

Three artifacts are persisted after training:
  - xgb_model.pkl   : Trained XGBoost classifier
  - scaler.pkl      : Fitted StandardScaler (must match training)
  - threshold.json  : Optimal decision threshold

These are loaded by the FastAPI service in Phase 2.
"""

import json
import logging
import os
import joblib

logger = logging.getLogger(__name__)

MODEL_FILENAME = "xgb_model.pkl"
SCALER_FILENAME = "scaler.pkl"
THRESHOLD_FILENAME = "threshold.json"


def save_artifacts(model, scaler, threshold: float, output_dir: str = "models") -> None:
    """
    Persist all three artifacts to disk.

    Args:
        model: Fitted XGBClassifier
        scaler: Fitted StandardScaler
        threshold: Optimal decision threshold (float)
        output_dir: Directory to write artifacts into
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, MODEL_FILENAME)
    scaler_path = os.path.join(output_dir, SCALER_FILENAME)
    threshold_path = os.path.join(output_dir, THRESHOLD_FILENAME)

    joblib.dump(model, model_path)
    logger.info(f"Model saved     → {model_path}")

    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved    → {scaler_path}")

    with open(threshold_path, "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    logger.info(f"Threshold saved → {threshold_path}  (value: {threshold})")


def load_artifacts(model_dir: str = "models"):
    """
    Load all three artifacts from disk.
    Called by the FastAPI ModelService at startup.

    Args:
        model_dir: Directory containing the artifact files

    Returns:
        model: Fitted XGBClassifier
        scaler: Fitted StandardScaler
        threshold: Float decision threshold

    Raises:
        FileNotFoundError: If any artifact file is missing.
    """
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    scaler_path = os.path.join(model_dir, SCALER_FILENAME)
    threshold_path = os.path.join(model_dir, THRESHOLD_FILENAME)

    for path in [model_path, scaler_path, threshold_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Run `python scripts/train_pipeline.py` first to generate model artifacts."
            )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

    logger.info(f"Model loaded from    : {model_path}")
    logger.info(f"Scaler loaded from   : {scaler_path}")
    logger.info(f"Threshold loaded     : {threshold} (from {threshold_path})")

    return model, scaler, threshold