"""
response.py
Pydantic schemas for API responses.
All response models are explicitly defined — no raw dicts returned from routes.
"""

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' if model is loaded and ready.")
    model_loaded: bool
    version: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }
    }


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    features: int
    report_available: bool
    roc_auc: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None


class PredictionResponse(BaseModel):
    is_fraud: bool = Field(..., description="True if the transaction is predicted as fraudulent.")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Raw fraud probability from the model.")
    threshold_used: float = Field(..., description="The decision threshold applied to produce is_fraud.")
    model_version: str
    latency_ms: float = Field(..., description="Inference time in milliseconds.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "is_fraud": False,
                "fraud_probability": 0.0314,
                "threshold_used": 0.88,
                "model_version": "1.0.0",
                "latency_ms": 4.7
            }
        }
    }


class BatchPredictionItem(BaseModel):
    index: int = Field(..., description="Position of this transaction in the input list (0-based).")
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    threshold_used: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    count: int = Field(..., description="Number of transactions processed.")
    results: list[BatchPredictionItem]