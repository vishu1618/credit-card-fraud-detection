"""
trainer.py
Trains Logistic Regression, Random Forest, and XGBoost classifiers.
Returns fitted model objects for comparison and selection.
"""

import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def train_logistic_regression(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """Train a Logistic Regression baseline model."""
    logger.info("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",  # handles remaining imbalance after SMOTE
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")
    return model


def train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    """Train a Random Forest baseline model."""
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return model


def train_xgboost(X_train, y_train, params: dict = None, random_state: int = 42) -> XGBClassifier:
    """
    Train an XGBoost classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        params: Optional dict of XGBoost hyperparameters.
                If None, uses sensible defaults.
        random_state: Seed for reproducibility

    Returns:
        Fitted XGBClassifier
    """
    logger.info("Training XGBoost...")

    default_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
    }

    if params:
        default_params.update(params)

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    logger.info("XGBoost training complete.")
    return model


def compare_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models on the test set and return a comparison table.

    Args:
        models: Dict of {model_name: fitted_model}
        X_test: Test features
        y_test: True test labels

    Returns:
        DataFrame with Precision, Recall, F1, ROC-AUC per model.
    """
    logger.info("Comparing models on test set...")
    results = []

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_proba, threshold=0.5)
        metrics["Model"] = name
        results.append(metrics)

    df = pd.DataFrame(results).set_index("Model")
    df = df[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]

    logger.info("\n" + df.round(4).to_string())
    return df