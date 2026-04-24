import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_utils import OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs, raw_data_path


def main() -> None:
    ensure_project_dirs()
    dataset_path = raw_data_path()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    sns.set_theme(style="darkgrid")
    df = pd.read_csv(dataset_path)

    print("Dataset shape:", df.shape)
    print("Null values:", int(df.isna().sum().sum()))

    expected_rows = 284_807
    expected_columns = 31
    if df.shape != (expected_rows, expected_columns):
        print(
            "Warning: expected shape "
            f"({expected_rows}, {expected_columns}) but found {df.shape}. Proceeding with available data."
        )

    class_counts = df["Class"].value_counts().sort_index()
    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]

    # 1) Class distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Legitimate", "Fraud"], [class_counts[0], class_counts[1]], color=["#2ca02c", "#d62728"])
    ax.set_yscale("log")
    ax.set_title("Class Distribution (Log Scale)")
    ax.set_ylabel("Count (log scale)")
    for idx, value in enumerate([class_counts[0], class_counts[1]]):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)

    # 2) Amount distribution fraud vs legit
    fig, ax = plt.subplots(figsize=(10, 5))
    max_amount = float(df["Amount"].quantile(0.995))
    sns.histplot(
        legit_df["Amount"],
        bins=120,
        stat="density",
        kde=True,
        color="#1f77b4",
        alpha=0.45,
        ax=ax,
        label="Legitimate",
    )
    sns.histplot(
        fraud_df["Amount"],
        bins=120,
        stat="density",
        kde=True,
        color="#d62728",
        alpha=0.55,
        ax=ax,
        label="Fraud",
    )
    ax.set_xlim(0, max_amount)
    ax.set_title("Transaction Amount Distribution: Fraud vs Legitimate")
    ax.set_xlabel("Amount (EUR)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "amount_distribution_fraud_vs_legit.png", dpi=200)
    plt.close(fig)

    # 3) Time distribution fraud vs legit
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(legit_df["Time"], label="Legitimate", fill=True, alpha=0.25, color="#1f77b4", ax=ax)
    sns.kdeplot(fraud_df["Time"], label="Fraud", fill=True, alpha=0.35, color="#d62728", ax=ax)
    ax.set_title("Time Distribution: Fraud vs Legitimate")
    ax.set_xlabel("Seconds Since First Transaction")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "time_distribution_fraud_vs_legit.png", dpi=200)
    plt.close(fig)

    # 4) Correlation with Class for V1-V28
    v_features = [f"V{i}" for i in range(1, 29)]
    corr_series = df[v_features + ["Class"]].corr()["Class"].drop("Class")
    corr_abs_sorted = corr_series.reindex(corr_series.abs().sort_values().index)

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["#d62728" if val < 0 else "#2ca02c" for val in corr_abs_sorted.values]
    ax.barh(corr_abs_sorted.index, corr_abs_sorted.values, color=colors)
    ax.set_title("Feature Correlation with Fraud Class (V1-V28)")
    ax.set_xlabel("Correlation with Class")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "feature_correlation.png", dpi=200)
    plt.close(fig)

    # 5) Box plots for top correlated features split by class
    top_features = ["V14", "V10", "V12", "V4", "V11", "V17"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    box_df = df.copy()
    box_df["Class"] = box_df["Class"].astype(str)
    for ax, feature in zip(axes.flatten(), top_features):
        sns.boxplot(
            data=box_df,
            x="Class",
            y=feature,
            hue="Class",
            ax=ax,
            palette={"0": "#1f77b4", "1": "#d62728"},
            legend=False,
        )
        ax.set_title(f"{feature} by Class")
        ax.set_xlabel("Class (0=Legitimate, 1=Fraud)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "top_feature_boxplots.png", dpi=200)
    plt.close(fig)

    corr_df = corr_series.sort_values(key=lambda s: s.abs(), ascending=False).rename_axis("feature").to_frame("corr")
    corr_df.to_csv(OUTPUTS_DIR / "feature_correlation_values.csv", index=True)

    feature_ranges = {}
    for feature in top_features:
        feature_ranges[feature] = {"min": float(df[feature].min()), "max": float(df[feature].max())}
    feature_ranges_path = PROCESSED_DATA_DIR / "feature_ranges.json"
    feature_ranges_path.write_text(json.dumps(feature_ranges, indent=2), encoding="utf-8")

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "null_values": int(df.isna().sum().sum()),
        "class_distribution": {"legitimate": int(class_counts[0]), "fraud": int(class_counts[1])},
        "fraud_rate_percent": round(float(class_counts[1] / len(df) * 100), 4),
        "median_amount": {
            "legitimate": round(float(legit_df["Amount"].median()), 4),
            "fraud": round(float(fraud_df["Amount"].median()), 4),
        },
        "top_correlated_features": corr_df.head(10)["corr"].to_dict(),
    }
    (OUTPUTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved EDA outputs to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
