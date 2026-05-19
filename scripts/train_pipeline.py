import sys
import os
import argparse
import logging
import time

# Make src/ importable regardless of where the script is called from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_dataset
from src.data.validator import run_all_validations
from src.features.engineering import separate_features_and_target, fit_scaler, apply_scaling
from src.training.splitter import stratified_split
from src.training.sampler import apply_smote
from src.training.trainer import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    compare_models,
)
from src.training.tuner import hyperparameter_search
from src.evaluation.metrics import compute_metrics, print_full_report, save_metrics_report
from src.evaluation.threshold import find_optimal_threshold
from src.persistence.model_io import save_artifacts

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_pipeline")

DATASET_PATH = "data/raw/creditcard.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the fraud detection ML pipeline.")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip GridSearchCV hyperparameter tuning (faster run, uses default XGBoost params)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print("\n" + "="*60)
    print("   CREDIT CARD FRAUD DETECTION — TRAINING PIPELINE")
    print("="*60 + "\n")

    # ── Step 1: Load & Validate ────────────────────────────────────────────────
    logger.info("STEP 1/9 — Loading dataset")
    df = load_dataset(DATASET_PATH)
    df = run_all_validations(df)

    # ── Step 2: Feature Engineering ───────────────────────────────────────────
    logger.info("STEP 2/9 — Separating features and target")
    X, y = separate_features_and_target(df)

    # ── Step 3: Train/Test Split (before scaling to avoid leakage) ────────────
    logger.info("STEP 3/9 — Stratified train/test split (80/20)")
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    # ── Step 4: Scaling (fit on train only, apply to both) ────────────────────
    logger.info("STEP 4/9 — Scaling Amount and Time")
    scaler = fit_scaler(X_train)
    X_train = apply_scaling(X_train, scaler)
    X_test = apply_scaling(X_test, scaler)

    # ── Step 5: SMOTE (only on training data) ─────────────────────────────────
    logger.info("STEP 5/9 — Applying SMOTE to training set")
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # ── Step 6: Train all models ───────────────────────────────────────────────
    logger.info("STEP 6/9 — Training baseline models")
    lr_model = train_logistic_regression(X_train_bal, y_train_bal)
    rf_model = train_random_forest(X_train_bal, y_train_bal)
    xgb_model = train_xgboost(X_train_bal, y_train_bal)

    # ── Step 7: Compare models ─────────────────────────────────────────────────
    logger.info("STEP 7/9 — Comparing models on test set")
    print("\n  Model Comparison (threshold=0.5, test set):")
    comparison_df = compare_models(
        models={
            "Logistic Regression": lr_model,
            "Random Forest": rf_model,
            "XGBoost": xgb_model,
        },
        X_test=X_test,
        y_test=y_test,
    )
    print()
    print(comparison_df.round(4).to_string())
    print()

    # ── Step 8: Hyperparameter tuning ─────────────────────────────────────────
    if args.skip_tuning:
        logger.info("STEP 8/9 — Skipping hyperparameter tuning (--skip-tuning flag set)")
        best_params = {}
    else:
        logger.info("STEP 8/9 — Tuning XGBoost hyperparameters")
        best_params = hyperparameter_search(X_train_bal, y_train_bal)
        logger.info("Re-training XGBoost with best parameters...")
        xgb_model = train_xgboost(X_train_bal, y_train_bal, params=best_params)

    # ── Step 9: Threshold optimization ────────────────────────────────────────
    logger.info("STEP 9/9 — Finding optimal classification threshold")
    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    optimal_threshold = find_optimal_threshold(
        y_true=y_test,
        y_proba=y_proba,
        min_precision=0.85,
    )

    # ── Final Evaluation ───────────────────────────────────────────────────────
    final_metrics = compute_metrics(y_test, y_proba, threshold=optimal_threshold)
    print_full_report(y_test, y_proba, threshold=optimal_threshold, model_name="XGBoost (Tuned)")

    # ── Save Artifacts ─────────────────────────────────────────────────────────
    logger.info("Saving model artifacts...")
    save_artifacts(
        model=xgb_model,
        scaler=scaler,
        threshold=optimal_threshold,
        output_dir=MODELS_DIR,
    )

    # ── Save metrics report ────────────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "model_report.json")
    save_metrics_report(
        metrics=final_metrics,
        comparison_df=comparison_df,
        output_path=report_path,
    )

    # ── Done ───────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("   PIPELINE COMPLETE")
    print("="*60)
    print(f"   Total time     : {elapsed:.1f}s")
    print(f"   Model saved    : {MODELS_DIR}/xgb_model.pkl")
    print(f"   Scaler saved   : {MODELS_DIR}/scaler.pkl")
    print(f"   Threshold      : {MODELS_DIR}/threshold.json  ({optimal_threshold})")
    print(f"   Report saved   : {report_path}")
    print(f"\n   ROC-AUC        : {final_metrics['ROC_AUC']}")
    print(f"   Recall         : {final_metrics['Recall']}  ← fraud catch rate")
    print(f"   F1 Score       : {final_metrics['F1']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
