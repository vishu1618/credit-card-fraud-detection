"""
metrics.py
Computes and formats evaluation metrics for fraud detection models.

For fraud detection, Recall is the most important metric —
missing a real fraud (false negative) is more costly than a false alarm.
"""

import logging
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """
    Compute all key classification metrics at a given threshold.

    Args:
        y_true: Ground truth labels (0/1)
        y_proba: Predicted probabilities for class 1 (fraud)
        threshold: Decision boundary (default 0.5)

    Returns:
        Dict with Accuracy, Precision, Recall, F1, ROC_AUC.
    """
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Threshold": threshold,
    }


def print_full_report(y_true, y_proba, threshold: float, model_name: str = "XGBoost") -> None:
    """
    Print a complete evaluation report: metrics table + confusion matrix.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities
        threshold: Decision threshold to apply
        model_name: Label for the report header
    """
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_proba, threshold)

    print(f"\n{'='*55}")
    print(f"  Final Evaluation Report — {model_name}")
    print(f"{'='*55}")
    print(f"  Threshold Used : {threshold:.2f}")
    print(f"  Accuracy       : {metrics['Accuracy']:.4f}  ({metrics['Accuracy']*100:.2f}%)")
    print(f"  Precision      : {metrics['Precision']:.4f}")
    print(f"  Recall         : {metrics['Recall']:.4f}  ← fraud catch rate")
    print(f"  F1 Score       : {metrics['F1']:.4f}")
    print(f"  ROC-AUC        : {metrics['ROC_AUC']:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"  {'':20} Predicted Legit  Predicted Fraud")
    print(f"  {'Actual Legit':20} {tn:>13,}  {fp:>14,}")
    print(f"  {'Actual Fraud':20} {fn:>13,}  {tp:>14,}  ← TP")
    print(f"\n  True Positives  (caught fraud):   {tp:,}")
    print(f"  False Negatives (missed fraud):   {fn:,}")
    print(f"  False Positives (false alarms):   {fp:,}")
    print(f"  True Negatives  (correct legit):  {tn:,}")
    print(f"{'='*55}\n")


def save_metrics_report(metrics: dict, comparison_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the final metrics and model comparison table to a JSON file.

    Args:
        metrics: Final XGBoost metrics dict
        comparison_df: DataFrame from compare_models()
        output_path: Where to write the JSON (e.g. reports/model_report.json)
    """
    report = {
        "final_model": "XGBoost",
        "metrics": metrics,
        "model_comparison": comparison_df.round(4).to_dict(),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Metrics report saved to: {output_path}")
