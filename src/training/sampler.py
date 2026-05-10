"""
sampler.py
Applies SMOTE to handle class imbalance on the training set ONLY.

SMOTE (Synthetic Minority Over-sampling Technique) generates new fraud
samples by interpolating between existing fraud cases — not by duplicating them.

CRITICAL: Never apply SMOTE to the test set. Doing so would contaminate
evaluation with synthetic data and produce misleadingly good metrics.
"""

import logging
import pandas as pd
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """
    Balance the training set by oversampling the fraud minority class with SMOTE.

    Args:
        X_train: Training features
        y_train: Training labels (must contain imbalanced fraud class)
        random_state: Seed for reproducibility

    Returns:
        X_resampled (pd.DataFrame): Balanced feature matrix
        y_resampled (pd.Series): Balanced target vector
    """
    logger.info(
        f"Before SMOTE — Legit: {(y_train == 0).sum():,} | "
        f"Fraud: {(y_train == 1).sum():,}"
    )

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Wrap back in DataFrame/Series to preserve column names
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    logger.info(
        f"After SMOTE  — Legit: {(y_res == 0).sum():,} | "
        f"Fraud: {(y_res == 1).sum():,}"
    )

    return X_res, y_res