"""
splitter.py
Stratified train/test split that preserves the fraud class ratio.
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets using stratification.
    Stratification ensures the ~0.17% fraud ratio is preserved in both halves.

    Args:
        X: Feature matrix
        y: Target vector (Class column)
        test_size: Fraction of data for testing (default 0.2 = 20%)
        random_state: Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # critical: preserves fraud ratio in both splits
    )

    logger.info(f"Train set: {len(X_train):,} rows | Test set: {len(X_test):,} rows")
    logger.info(
        f"Train fraud count: {y_train.sum():,} ({y_train.mean() * 100:.3f}%) | "
        f"Test fraud count: {y_test.sum():,} ({y_test.mean() * 100:.3f}%)"
    )

    return X_train, X_test, y_train, y_test