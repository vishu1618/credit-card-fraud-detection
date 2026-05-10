"""
loader.py
Loads the raw CSV dataset into a validated Pandas DataFrame.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = (
    [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
)


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the credit card fraud CSV from disk.

    Args:
        path: Path to creditcard.csv

    Returns:
        Raw DataFrame with all original columns intact.

    Raises:
        FileNotFoundError: If the file does not exist at path.
        ValueError: If expected columns are missing.
    """
    logger.info(f"Loading dataset from: {path}")

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it at data/raw/creditcard.csv"
        )

    logger.info(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    fraud_count = df["Class"].sum()
    fraud_pct = fraud_count / len(df) * 100
    logger.info(
        f"Class distribution — Legitimate: {len(df) - fraud_count:,} | "
        f"Fraud: {fraud_count:,} ({fraud_pct:.3f}%)"
    )

    return df