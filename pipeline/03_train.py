import json
import sys
from pathlib import Path

import joblib
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


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_utils import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def main() -> None:
    ensure_project_dirs()

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
    iso_model = IsolationForest(
        n_estimators=200,
        contamination=0.0017,
        random_state=42,
        n_jobs=1,
    )
    iso_model.fit(X_train)
    iso_pred = (iso_model.predict(X_test) == -1).astype(int)
    iso_score = -iso_model.decision_function(X_test)
    metrics["IsolationForest"] = compute_metrics(y_test, iso_pred, iso_score)
    scores["IsolationForest"] = iso_score

    # 2) One-Class SVM baseline (fit on legit only, subsampled)
    legit_only = X_train[y_train == 0]
    legit_subset_size = min(50_000, len(legit_only))
    legit_subset = legit_only.sample(n=legit_subset_size, random_state=42)
    ocsvm_model = OneClassSVM(kernel="rbf", nu=fraud_rate, gamma="scale")
    ocsvm_model.fit(legit_subset)
    ocsvm_pred = (ocsvm_model.predict(X_test) == -1).astype(int)
    ocsvm_score = -ocsvm_model.decision_function(X_test)
    metrics["OneClassSVM"] = compute_metrics(y_test, ocsvm_pred, ocsvm_score)
    scores["OneClassSVM"] = ocsvm_score

    # Apply SMOTE once for supervised model training artifacts
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    smote_summary = {
        "before_smote": {
            "legitimate": int((y_train == 0).sum()),
            "fraud": int((y_train == 1).sum()),
            "fraud_ratio": float((y_train == 1).mean()),
        },
        "after_smote": {
            "legitimate": int((y_train_sm == 0).sum()),
            "fraud": int((y_train_sm == 1).sum()),
            "fraud_ratio": float((y_train_sm == 1).mean()),
        },
    }
    (OUTPUTS_DIR / "smote_summary.json").write_text(json.dumps(smote_summary, indent=2), encoding="utf-8")

    # 3) Random Forest + SMOTE
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
        class_weight="balanced",
    )
    rf_model.fit(X_train_sm, y_train_sm)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    metrics["RandomForest_SMOTE"] = compute_metrics(y_test, rf_pred, rf_proba)
    scores["RandomForest_SMOTE"] = rf_proba

    # 4) XGBoost + SMOTE + threshold tuning (leakage-safe)
    # Split train into fit/validation so threshold is not tuned on the test set.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    xgb_pipeline = ImbPipeline(
        steps=[
            ("smote", SMOTE(sampling_strategy=0.1, random_state=42)),
            ("xgb", xgb_base),
        ]
    )
    param_distributions = {
        "xgb__n_estimators": [200, 300, 400],
        "xgb__max_depth": [3, 4, 5, 6],
        "xgb__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
        "xgb__scale_pos_weight": [300, 400, 500, 578, 700],
    }
    search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=param_distributions,
        n_iter=6,
        scoring="f1",
        cv=3,
        random_state=42,
        verbose=1,
        n_jobs=1,
    )
    search.fit(X_fit, y_fit)

    val_proba = search.best_estimator_.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 161)
    threshold_rows = []
    for threshold in thresholds:
        threshold_pred = (val_proba >= threshold).astype(int)
        precision = precision_score(y_val, threshold_pred, zero_division=0)
        recall = recall_score(y_val, threshold_pred, zero_division=0)
        f1 = f1_score(y_val, threshold_pred, zero_division=0)
        threshold_rows.append((float(threshold), float(precision), float(recall), float(f1)))
    threshold_df = pd.DataFrame(threshold_rows, columns=["threshold", "precision", "recall", "f1"])
    threshold_df.to_csv(OUTPUTS_DIR / "xgb_threshold_metrics.csv", index=False)

    # Instruction-specific operating range for fraud detection.
    operating_range = threshold_df[(threshold_df["threshold"] >= 0.2) & (threshold_df["threshold"] <= 0.3)]
    if len(operating_range) > 0:
        best_row = operating_range.loc[operating_range["f1"].idxmax()]
    else:
        best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    best_threshold = float(best_row["threshold"])
    global_best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    global_best_threshold = float(global_best_row["threshold"])

    # Refit best params on full training set for final model.
    best_xgb_params = {k.replace("xgb__", ""): v for k, v in search.best_params_.items() if k.startswith("xgb__")}
    final_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
        **best_xgb_params,
    )
    xgb_model = ImbPipeline(
        steps=[
            ("smote", SMOTE(sampling_strategy=0.1, random_state=42)),
            ("xgb", final_xgb),
        ]
    )
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_proba >= best_threshold).astype(int)
    metrics["XGBoost_SMOTE_Tuned"] = compute_metrics(y_test, xgb_pred, xgb_proba)
    scores["XGBoost_SMOTE_Tuned"] = xgb_proba

    # Save trained models
    joblib.dump(iso_model, MODELS_DIR / "isolation_forest.pkl")
    joblib.dump(ocsvm_model, MODELS_DIR / "one_class_svm.pkl")
    joblib.dump(rf_model, MODELS_DIR / "random_forest_smote.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_fraud.pkl")

    model_meta = {
        "xgb_best_params": search.best_params_,
        "xgb_best_cv_f1": float(search.best_score_),
        "xgb_best_threshold": best_threshold,
        "xgb_global_best_threshold": global_best_threshold,
        "threshold_source": "validation_split",
        "feature_columns": X_train.columns.tolist(),
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")
    (OUTPUTS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    scores_df = pd.DataFrame(
        {
            "y_true": scores["y_true"],
            "IsolationForest_score": scores["IsolationForest"],
            "OneClassSVM_score": scores["OneClassSVM"],
            "RandomForest_SMOTE_score": scores["RandomForest_SMOTE"],
            "XGBoost_SMOTE_Tuned_score": scores["XGBoost_SMOTE_Tuned"],
        }
    )
    scores_df.to_csv(OUTPUTS_DIR / "test_scores.csv", index=False)

    print("Training complete. Metrics:")
    for model_name, model_metrics in metrics.items():
        print(model_name, model_metrics)
    print("XGBoost operating threshold (0.2-0.3 range):", best_threshold)
    print("XGBoost global best threshold on validation:", global_best_threshold)


if __name__ == "__main__":
    main()
