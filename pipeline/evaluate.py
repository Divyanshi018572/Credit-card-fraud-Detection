import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_utils import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


def model_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "PR_AUC": float(average_precision_score(y_true, y_score)),
        "ROC_AUC": float(roc_auc_score(y_true, y_score)),
    }


def main() -> None:
    ensure_project_dirs()
    sns.set_theme(style="darkgrid")

    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    model_meta = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]
    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    iso_model = joblib.load(MODELS_DIR / "isolation_forest.pkl")
    ocsvm_model = joblib.load(MODELS_DIR / "one_class_svm.pkl")
    rf_model = joblib.load(MODELS_DIR / "random_forest_smote.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgboost_fraud.pkl")

    iso_score = -iso_model.decision_function(X_test)
    ocsvm_score = -ocsvm_model.decision_function(X_test)
    rf_score = rf_model.predict_proba(X_test)[:, 1]
    xgb_score = xgb_model.predict_proba(X_test)[:, 1]

    xgb_threshold = float(model_meta["xgb_best_threshold"])
    iso_pred = (iso_model.predict(X_test) == -1).astype(int)
    ocsvm_pred = (ocsvm_model.predict(X_test) == -1).astype(int)
    rf_pred = (rf_score >= 0.5).astype(int)
    xgb_pred = (xgb_score >= xgb_threshold).astype(int)

    metrics_by_model = {
        "Isolation Forest": model_metrics(y_test, iso_pred, iso_score),
        "One-Class SVM": model_metrics(y_test, ocsvm_pred, ocsvm_score),
        "Random Forest + SMOTE": model_metrics(y_test, rf_pred, rf_score),
        "XGBoost + SMOTE + Threshold": model_metrics(y_test, xgb_pred, xgb_score),
    }

    comparison_df = pd.DataFrame(metrics_by_model).T
    comparison_df = comparison_df[["F1", "Recall", "Precision", "PR_AUC", "ROC_AUC"]]
    comparison_df.to_csv(OUTPUTS_DIR / "model_comparison_table.csv", index=True)

    # PR curve for all models (primary)
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, score, color in [
        ("Isolation Forest", iso_score, "#ff7f0e"),
        ("One-Class SVM", ocsvm_score, "#9467bd"),
        ("Random Forest + SMOTE", rf_score, "#1f77b4"),
        ("XGBoost + SMOTE + Threshold", xgb_score, "#2ca02c"),
    ]:
        precision, recall, _ = precision_recall_curve(y_test, score)
        pr_auc = average_precision_score(y_test, score)
        ax.plot(recall, precision, linewidth=2, color=color, label=f"{label} (PR-AUC={pr_auc:.3f})")
    random_baseline = y_test.mean()
    ax.axhline(y=random_baseline, color="#d62728", linestyle="--", label=f"Random baseline ({random_baseline:.4f})")
    ax.set_title("Precision-Recall Curves: All Models")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "pr_curve_all_models.png", dpi=220)
    plt.close(fig)

    # Threshold optimization chart for XGBoost
    threshold_df = pd.read_csv(OUTPUTS_DIR / "xgb_threshold_metrics.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1", color="#2ca02c", linewidth=2)
    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", color="#1f77b4", linewidth=2)
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", color="#d62728", linewidth=2)
    ax.axvline(x=xgb_threshold, linestyle="--", color="black", label=f"Best Threshold={xgb_threshold:.3f}")
    ax.set_title("XGBoost Threshold Optimization")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "threshold_optimization.png", dpi=220)
    plt.close(fig)

    # Confusion matrix (XGBoost at tuned threshold)
    cm = confusion_matrix(y_test, xgb_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix - XGBoost (threshold={xgb_threshold:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_yticklabels(["Legitimate", "Fraud"], rotation=0)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "confusion_matrix_xgb.png", dpi=220)
    plt.close(fig)

    # SMOTE impact visualization (before vs after)
    smote_summary = json.loads((OUTPUTS_DIR / "smote_summary.json").read_text(encoding="utf-8"))
    before_legit = smote_summary["before_smote"]["legitimate"]
    before_fraud = smote_summary["before_smote"]["fraud"]
    after_legit = smote_summary["after_smote"]["legitimate"]
    after_fraud = smote_summary["after_smote"]["fraud"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].bar(["Legitimate", "Fraud"], [before_legit, before_fraud], color=["#1f77b4", "#d62728"])
    axes[0].set_title("Before SMOTE")
    axes[0].set_yscale("log")
    axes[1].bar(["Legitimate", "Fraud"], [after_legit, after_fraud], color=["#1f77b4", "#d62728"])
    axes[1].set_title("After SMOTE (10% fraud ratio)")
    axes[1].set_yscale("log")
    fig.suptitle("Class Distribution Before vs After SMOTE")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "smote_impact.png", dpi=220)
    plt.close(fig)

    # F1 score vs SMOTE sampling strategy
    strategies = [0.05, 0.1, 0.2, 0.5]
    strategy_scores = []
    for strategy in strategies:
        smote = SMOTE(sampling_strategy=strategy, random_state=42)
        X_sm, y_sm = smote.fit_resample(X_train, y_train)
        rf = RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            n_jobs=1,
            class_weight="balanced",
        )
        rf.fit(X_sm, y_sm)
        pred = rf.predict(X_test)
        strategy_scores.append(float(f1_score(y_test, pred, zero_division=0)))

    strategy_df = pd.DataFrame({"sampling_strategy": strategies, "f1_score": strategy_scores})
    strategy_df.to_csv(OUTPUTS_DIR / "smote_strategy_f1.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(strategy_df["sampling_strategy"], strategy_df["f1_score"], marker="o", linewidth=2, color="#2ca02c")
    ax.set_title("F1 Score vs SMOTE Sampling Strategy")
    ax.set_xlabel("SMOTE sampling_strategy (fraud/legit ratio)")
    ax.set_ylabel("F1 Score (Fraud)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "f1_vs_smote_sampling_strategy.png", dpi=220)
    plt.close(fig)

    # Naive vs SMOTE for educational chart in app
    naive_rf = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=1)
    naive_rf.fit(X_train, y_train)
    naive_pred = naive_rf.predict(X_test)
    naive_f1 = float(f1_score(y_test, naive_pred, zero_division=0))
    smote_f1 = float(metrics_by_model["XGBoost + SMOTE + Threshold"]["F1"])
    pd.DataFrame(
        {"Model": ["Naive RF (No SMOTE)", "Best Pipeline (XGBoost+SMOTE+Threshold)"], "F1": [naive_f1, smote_f1]}
    ).to_csv(OUTPUTS_DIR / "naive_vs_smote_metrics.csv", index=False)

    # Save combined test scores for app confusion matrix updates
    pd.DataFrame(
        {
            "y_true": y_test,
            "Isolation Forest": (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-9),
            "One-Class SVM": (ocsvm_score - ocsvm_score.min()) / (ocsvm_score.max() - ocsvm_score.min() + 1e-9),
            "Random Forest + SMOTE": rf_score,
            "XGBoost + SMOTE + Threshold": xgb_score,
        }
    ).to_csv(OUTPUTS_DIR / "model_test_probabilities.csv", index=False)

    best_f1_model = max(metrics_by_model.items(), key=lambda item: item[1]["F1"])[0]
    best_pr_auc_model = max(metrics_by_model.items(), key=lambda item: item[1]["PR_AUC"])[0]
    summary = {
        "best_model_by_f1": best_f1_model,
        "best_model_by_pr_auc": best_pr_auc_model,
        "xgb_threshold": xgb_threshold,
        "metrics": metrics_by_model,
    }
    (OUTPUTS_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Evaluation complete. Outputs saved in:", OUTPUTS_DIR)
    print(comparison_df)


if __name__ == "__main__":
    main()
