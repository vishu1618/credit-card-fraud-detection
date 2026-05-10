"""
validator.py
Data quality checks run before any preprocessing begins.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = (
    [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
)


def validate_schema(df: pd.DataFrame) -> None:
    """
    Confirm all 31 expected columns are present.

    Raises:
        ValueError: If any expected column is missing.
    """
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}\n"
            "Ensure you are using the Kaggle ULB creditcard.csv dataset."
        )
    logger.info("Schema validation passed — all 31 columns present.")


def validate_no_nulls(df: pd.DataFrame) -> None:
    """
    Confirm there are no missing values in the dataset.

    Raises:
        ValueError: If any column contains null values.
    """
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if not cols_with_nulls.empty:
        raise ValueError(
            f"Dataset contains null values:\n{cols_with_nulls}"
        )
    logger.info("Null check passed — no missing values found.")


def validate_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows if any exist.

    Returns:
        DataFrame with duplicates removed.
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    if removed > 0:
        logger.warning(f"Removed {removed:,} duplicate rows.")
    else:
        logger.info("Duplicate check passed — no duplicates found.")

    return df


def run_all_validations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all data quality checks in sequence.
    Call this once after loading the raw dataset.

    Returns:
        Cleaned DataFrame ready for preprocessing.
    """
    logger.info("Running data validation checks...")
    validate_schema(df)
    validate_no_nulls(df)
    df = validate_and_deduplicate(df)
    logger.info("All validation checks passed.")
    return df