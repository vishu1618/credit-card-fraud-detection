# Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange?logo=xgboost)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9747-brightgreen)
![Recall](https://img.shields.io/badge/Recall-77.89%25-blue)

> End-to-end fraud detection system trained on 284,807 real credit card transactions.
> Detects fraud with **97.47% ROC-AUC** and serves predictions via a production-style
> FastAPI REST API with full Swagger documentation.

---

## Live Demo

| Endpoint | Description |
|---|---|
| `GET /health` | API liveness check |
| `GET /model-info` | Model metadata + training metrics |
| `POST /predict` | Single transaction fraud prediction |
| `POST /batch-predict` | Batch predictions (up to 100 transactions) |

> Swagger UI: [localhost:8000/docs](http://localhost:8000/docs) when running locally

---

## Why This Project

Credit card fraud costs the industry $32B+ annually. Standard classifiers fail on fraud data because genuine fraud is only **0.17% of transactions** — a model that predicts "not fraud" for everything gets 99.8% accuracy while catching zero fraud cases.

This project solves that correctly:
- **SMOTE** generates synthetic fraud samples on the training set only (no leakage)
- **Threshold tuning** replaces the default 0.5 cutoff with an optimized value
- **Three models compared** — not just the first one that worked
- **FastAPI backend** serves the model as a real REST API, not a notebook

---

## Model Performance

Trained and evaluated on the [Kaggle ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud cases (0.173%).

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.053 | 0.874 | 0.100 | 0.962 |
| Random Forest | 0.911 | 0.758 | 0.828 | 0.966 |
| **XGBoost (tuned)** | **0.841** | **0.779** | **0.809** | **0.975** |

**Key decisions:**
- Decision threshold tuned to **0.88** (default 0.5 is wrong for imbalanced fraud data — it maximises accuracy, not recall)
- SMOTE applied to **training set only** — applying it before splitting is a data leakage bug that inflates test metrics
- GridSearchCV over 16 parameter combinations, 3-fold CV, scored on ROC-AUC

---

## Architecture




























---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost 2.0.3 |
| Preprocessing | scikit-learn, imbalanced-learn (SMOTE) |
| Data | Pandas, NumPy |
| API Framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Serialization | Joblib |
| Language | Python 3.11 |

---

## Project Structure













---

## Quick Start

### Prerequisites
- Python 3.11+
- Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`

### Install

```bash
git clone https://github.com/vishu1618/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

### Train the Model

```bash
# Full run with hyperparameter tuning (~3 min)
python scripts/train_pipeline.py

# Fast run, skip tuning (~30 sec)
python scripts/train_pipeline.py --skip-tuning
```

Expected output:














### Start the API

```bash
uvicorn app.main:app --reload
```

API is now running at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

---

## API Usage

### Check health

```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Get model info

```bash
curl http://localhost:8000/model-info
```
```json
{
  "model_name": "XGBoost Fraud Detector",
  "threshold": 0.88,
  "features": 30,
  "roc_auc": 0.9747,
  "recall": 0.7789,
  "f1": 0.8087
}
```

### Predict a single transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.3598071336738, "V2": -0.0727811733098497,
    "V3": 2.53634673796914, "V4": 1.37815522427443,
    "V5": -0.338320769942518, "V6": 0.462387777762292,
    "V7": 0.239598554061257, "V8": 0.0986979012610507,
    "V9": 0.363786969611213, "V10": 0.0907941719789316,
    "V11": -0.551599533260813, "V12": -0.617800855762348,
    "V13": -0.991389847235408, "V14": -0.311169353699879,
    "V15": 1.46817697209427, "V16": -0.470400525259478,
    "V17": 0.207971241929242, "V18": 0.0257905801985591,
    "V19": 0.403992960255733, "V20": 0.251412098239705,
    "V21": -0.018306777944153, "V22": 0.277837575558899,
    "V23": -0.110473910188767, "V24": 0.0669280749146731,
    "V25": 0.128539358273528, "V26": -0.189114843888824,
    "V27": 0.133558376740387, "V28": -0.0210530534538215,
    "Amount": 149.62, "Time": 0.0
  }'
```
```json
{
  "is_fraud": false,
  "fraud_probability": 0.000139,
  "threshold_used": 0.88,
  "model_version": "1.0.0",
  "latency_ms": 4.7
}
```

---

## Validation

The API rejects malformed requests with a `422 Unprocessable Entity`:

```bash
# Missing field → 422
# Negative Amount → 422
# Wrong type → 422
```

---

## Future Improvements

- [ ] Deploy to Render/Railway (live public API URL)
- [ ] GitHub Actions CI — auto-run tests on every push
- [ ] Docker containerization
- [ ] SHAP explainability — show which features drove each prediction
- [ ] LSTM model for temporal transaction sequence patterns
- [ ] Structured prediction logging for monitoring dashboard

---

## Dataset

[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle

- 284,807 transactions over 2 days
- 492 fraud cases (0.173%)
- Features V1–V28 are PCA-transformed (anonymized for confidentiality)
- Raw features: `Time` and `Amount`

The dataset is **not included** in this repository (Kaggle license). Download and place at `data/raw/creditcard.csv`.

---

## License

MIT — see [LICENSE](LICENSE)
