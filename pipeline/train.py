import json
import sys
from pathlib import Path
from typing import Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

# Set up project root for imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_utils import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs
from src.utils.config_loader import load_config


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Computes standard classification metrics for fraud detection.
    
    Args:
        y_true (pd.Series): True binary labels.
        y_pred (np.ndarray): Predicted binary labels.
        y_score (np.ndarray): Target scores (probabilities or decision function).
        
    Returns:
        Dict[str, float]: Dictionary containing Precision, Recall, F1, ROC_AUC, and PR_AUC.
    """
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    # ROC-AUC and PR-AUC require both classes to be present in y_true
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
        
    return metrics


def main() -> None:
    """
    Main training pipeline with MLflow tracking.
    """
    ensure_project_dirs()
    
    # Load project-wide configuration
    config = load_config()
    
    # Set MLflow experiment
    mlflow.set_experiment("Fraud_Detection_Optimization")
    
    with mlflow.start_run(run_name="Full_Pipeline_Run"):
        # Log all configuration parameters
        mlflow.log_dict(config, "config.yaml")
        
        # Load datasets
        train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
        test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

        X_train = train_df.drop(columns=["Class"])
        y_train = train_df["Class"]
        X_test = test_df.drop(columns=["Class"])
        y_test = test_df["Class"]

        fraud_rate = float(y_train.mean())
        metrics = {}
        scores = {"y_true": y_test.to_numpy()}

        # 1) Isolation Forest baseline
        iso_params = config["models"]["isolation_forest"]
        iso_model = IsolationForest(**iso_params)
        iso_model.fit(X_train)
        iso_pred = (iso_model.predict(X_test) == -1).astype(int)
        iso_score = -iso_model.decision_function(X_test)
        iso_metrics = compute_metrics(y_test, iso_pred, iso_score)
        metrics["IsolationForest"] = iso_metrics
        scores["IsolationForest"] = iso_score
        
        # Log IF metrics
        for k, v in iso_metrics.items():
            mlflow.log_metric(f"if_{k}", v)

        # 2) One-Class SVM baseline
        ocsvm_params = config["models"]["one_class_svm"]
        subsample_size = ocsvm_params.pop("subsample_size", 50000)
        legit_only = X_train[y_train == 0]
        legit_subset_size = min(subsample_size, len(legit_only))
        legit_subset = legit_only.sample(n=legit_subset_size, random_state=config["preprocessing"]["random_state"])
        
        ocsvm_model = OneClassSVM(**ocsvm_params)
        ocsvm_model.fit(legit_subset)
        ocsvm_pred = (ocsvm_model.predict(X_test) == -1).astype(int)
        ocsvm_score = -ocsvm_model.decision_function(X_test)
        ocsvm_metrics = compute_metrics(y_test, ocsvm_pred, ocsvm_score)
        metrics["OneClassSVM"] = ocsvm_metrics
        scores["OneClassSVM"] = ocsvm_score
        
        # Log OCSVM metrics
        for k, v in ocsvm_metrics.items():
            mlflow.log_metric(f"ocsvm_{k}", v)

        # Apply SMOTE
        smote_params = config["preprocessing"]["smote"]
        smote = SMOTE(**smote_params)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        # 3) Random Forest + SMOTE
        rf_params = config["models"]["random_forest"]
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train_sm, y_train_sm)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        rf_pred = (rf_proba >= 0.5).astype(int)
        rf_metrics = compute_metrics(y_test, rf_pred, rf_proba)
        metrics["RandomForest_SMOTE"] = rf_metrics
        scores["RandomForest_SMOTE"] = rf_proba
        
        # Log RF metrics
        for k, v in rf_metrics.items():
            mlflow.log_metric(f"rf_{k}", v)

        # 4) XGBoost + SMOTE + threshold tuning
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=config["preprocessing"]["test_size"],
            random_state=config["preprocessing"]["random_state"],
            stratify=y_train,
        )

        xgb_params = config["models"]["xgboost"]
        xgb_base = XGBClassifier(**xgb_params)
        xgb_pipeline = ImbPipeline(steps=[("smote", SMOTE(**smote_params)), ("xgb", xgb_base)])
        
        tuning_config = config["models"]["tuning"]
        param_distributions = {f"xgb__{k}": v for k, v in tuning_config["param_distributions"].items()}
        search = RandomizedSearchCV(
            estimator=xgb_pipeline,
            param_distributions=param_distributions,
            n_iter=tuning_config["n_iter"],
            cv=tuning_config["cv"],
            scoring="f1",
            random_state=tuning_config["random_state"],
            verbose=0,
            n_jobs=1,
        )
        search.fit(X_fit, y_fit)

        val_proba = search.best_estimator_.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 161)
        best_threshold = 0.5
        max_f1 = 0
        
        for t in thresholds:
            f1 = f1_score(y_val, (val_proba >= t).astype(int), zero_division=0)
            if f1 > max_f1:
                max_f1 = f1
                best_threshold = t
        
        # Log Best Threshold as a parameter
        mlflow.log_param("xgb_best_threshold", best_threshold)

        # Refit final XGBoost
        best_xgb_params = {k.replace("xgb__", ""): v for k, v in search.best_params_.items() if k.startswith("xgb__")}
        final_xgb = XGBClassifier(**{**xgb_params, **best_xgb_params})
        xgb_model = ImbPipeline(steps=[("smote", SMOTE(**smote_params)), ("xgb", final_xgb)])
        xgb_model.fit(X_train, y_train)
        
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred = (xgb_proba >= best_threshold).astype(int)
        xgb_metrics = compute_metrics(y_test, xgb_pred, xgb_proba)
        metrics["XGBoost_SMOTE_Tuned"] = xgb_metrics
        scores["XGBoost_SMOTE_Tuned"] = xgb_proba
        
        # Log XGB metrics
        for k, v in xgb_metrics.items():
            mlflow.log_metric(f"xgb_{k}", v)

        # Save artifacts locally
        joblib.dump(iso_model, MODELS_DIR / "isolation_forest.pkl")
        joblib.dump(ocsvm_model, MODELS_DIR / "one_class_svm.pkl")
        joblib.dump(rf_model, MODELS_DIR / "random_forest_smote.pkl")
        joblib.dump(xgb_model, MODELS_DIR / "xgboost_fraud.pkl")

        # Log models to MLflow
        mlflow.sklearn.log_model(rf_model, "random_forest_model")
        mlflow.sklearn.log_model(xgb_model, "xgboost_model")
        
        print(f"Training complete. MLflow Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
