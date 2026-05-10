"""
engineering.py
Feature engineering: scale Amount and Time, separate X and y.

The scaler is fitted ONLY on training data and reused at inference.
This file is the single source of truth for feature transformation.
"""

import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def separate_features_and_target(df: pd.DataFrame):
    """
    Split the DataFrame into feature matrix X and target vector y.

    Returns:
        X (pd.DataFrame): All columns except 'Class'
        y (pd.Series): The 'Class' column (0 = legit, 1 = fraud)
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]
    logger.info(f"Features: {X.shape[1]} columns | Target: 'Class'")
    return X, y


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on the Amount and Time columns of the training set.
    IMPORTANT: Only call this on training data — never on test data.

    Returns:
        Fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[["Amount", "Time"]])
    logger.info("StandardScaler fitted on 'Amount' and 'Time' columns.")
    return scaler


def apply_scaling(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to Amount and Time columns.
    Replaces the originals with scaled versions in-place.

    Args:
        X: Feature DataFrame (must contain 'Amount' and 'Time')
        scaler: A FITTED StandardScaler (from fit_scaler)

    Returns:
        DataFrame with Amount and Time replaced by their scaled versions.
    """
    X = X.copy()
    X[["Amount", "Time"]] = scaler.transform(X[["Amount", "Time"]])
    logger.info("Scaling applied to 'Amount' and 'Time'.")
    return X