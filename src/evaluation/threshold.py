"""
threshold.py
Finds the optimal classification threshold for fraud detection.

Default 0.5 is almost always wrong for imbalanced fraud data.
We sweep thresholds and find the one that maximizes Recall
while keeping Precision above a minimum acceptable floor.

Why this matters for fraud:
  - Missing fraud (low Recall) = real financial loss
  - Too many false alarms (low Precision) = customer frustration
  - We optimize for high Recall with a Precision floor of >= 0.85
"""

import logging
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_true,
    y_proba,
    min_precision: float = 0.85,
    sweep_start: float = 0.1,
    sweep_end: float = 0.9,
    sweep_step: float = 0.01,
) -> float:
    """
    Sweep decision thresholds and return the one that maximizes Recall
    subject to Precision >= min_precision.

    If no threshold satisfies the precision floor, falls back to the
    threshold that maximizes the F1 score.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted fraud probabilities
        min_precision: Minimum acceptable precision (default 0.85)
        sweep_start: Start of threshold sweep
        sweep_end: End of threshold sweep
        sweep_step: Step size for sweep

    Returns:
        Optimal threshold as a float.
    """
    thresholds = np.arange(sweep_start, sweep_end, sweep_step)

    best_threshold = 0.5
    best_recall = 0.0
    best_f1 = 0.0
    fallback_threshold = 0.5

    logger.info(
        f"Sweeping thresholds from {sweep_start} to {sweep_end} "
        f"(step={sweep_step}, min_precision={min_precision})..."
    )

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        # Skip if model predicts all one class (degenerate)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Track best F1 as fallback
        if f1 > best_f1:
            best_f1 = f1
            fallback_threshold = round(t, 2)

        # Primary objective: maximize Recall with Precision >= floor
        if precision >= min_precision and recall > best_recall:
            best_recall = recall
            best_threshold = round(t, 2)

    if best_recall > 0.0:
        logger.info(
            f"Optimal threshold found: {best_threshold} "
            f"(Recall={best_recall:.4f}, Precision >= {min_precision})"
        )
        return best_threshold
    else:
        logger.warning(
            f"No threshold achieved Precision >= {min_precision}. "
            f"Falling back to best F1 threshold: {fallback_threshold} (F1={best_f1:.4f})"
        )
        return fallback_threshold