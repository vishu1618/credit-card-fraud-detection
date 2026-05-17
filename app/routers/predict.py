"""
predict.py
Prediction endpoints.

POST /predict        — single transaction fraud prediction
POST /batch-predict  — batch prediction (up to 100 transactions)

Routers are intentionally thin — all logic lives in ModelService.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_model_service
from app.schemas.request import TransactionFeatures, BatchPredictionRequest
from app.schemas.response import PredictionResponse, BatchPredictionResponse, BatchPredictionItem
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    transaction: TransactionFeatures,
    service: ModelService = Depends(get_model_service),
):
    """
    Predict whether a single transaction is fraudulent.

    Provide all 30 features (V1-V28, Amount, Time).
    Returns fraud probability, binary prediction, and the threshold used.
    """
    try:
        result = service.predict_one(transaction.model_dump())
        logger.info(
            f"Prediction: is_fraud={result['is_fraud']} "
            f"probability={result['fraud_probability']:.4f} "
            f"latency={result['latency_ms']}ms"
        )
        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.")


@router.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(
    body: BatchPredictionRequest,
    service: ModelService = Depends(get_model_service),
):
    """
    Predict fraud for a batch of transactions (max 100).

    Each result includes its original index so you can map predictions
    back to your input list even if you process results out of order.
    """
    try:
        raw_transactions = [t.model_dump() for t in body.transactions]
        results = service.predict_batch(raw_transactions)
        logger.info(f"Batch prediction complete: {len(results)} transactions processed.")
        return BatchPredictionResponse(
            count=len(results),
            results=[BatchPredictionItem(**r) for r in results],
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed. Check server logs.")