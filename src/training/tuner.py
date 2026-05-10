"""
tuner.py
Lightweight hyperparameter search for XGBoost using GridSearchCV.
Uses ROC-AUC as the scoring metric (appropriate for imbalanced data).
"""

import logging
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Lightweight grid — covers the most impactful parameters without
# taking hours to run. Expand this grid for a more thorough search.
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}


def hyperparameter_search(X_train, y_train, cv: int = 3, random_state: int = 42) -> dict:
    """
    Run a grid search over XGBoost hyperparameters.

    Args:
        X_train: Training features (post-SMOTE)
        y_train: Training labels (post-SMOTE, balanced)
        cv: Number of cross-validation folds
        random_state: Seed for reproducibility

    Returns:
        best_params (dict): Parameters that achieved the highest ROC-AUC.
    """
    logger.info(f"Starting hyperparameter search (cv={cv}, grid size={_grid_size()})...")
    logger.info("This may take a few minutes...")

    base_model = XGBClassifier(
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    logger.info(f"Best ROC-AUC (CV): {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")

    return search.best_params_


def _grid_size() -> int:
    """Calculate total number of parameter combinations."""
    size = 1
    for values in PARAM_GRID.values():
        size *= len(values)
    return size