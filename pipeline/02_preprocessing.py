import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_utils import MODELS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs, raw_data_path


def main() -> None:
    ensure_project_dirs()
    df = pd.read_csv(raw_data_path())

    df["log_Amount"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Amount", "Time"])

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["log_Amount"] = scaler.fit_transform(X_train[["log_Amount"]])
    X_test["log_Amount"] = scaler.transform(X_test[["log_Amount"]])

    # Create full preprocessed file using training-fitted scaler parameters.
    X_all_scaled = X.copy()
    X_all_scaled["log_Amount"] = scaler.transform(X_all_scaled[["log_Amount"]])
    features_preprocessed = pd.concat([X_all_scaled, y], axis=1)
    features_preprocessed.to_csv(PROCESSED_DATA_DIR / "features_preprocessed.csv", index=False)

    train_df = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True).rename("Class")], axis=1
    )
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True).rename("Class")], axis=1)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    joblib.dump(scaler, MODELS_DIR / "log_amount_scaler.pkl")

    test_fraud_count = int(y_test.sum())
    metadata = {
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "train_class_distribution": {"legitimate": int((y_train == 0).sum()), "fraud": int((y_train == 1).sum())},
        "test_class_distribution": {"legitimate": int((y_test == 0).sum()), "fraud": test_fraud_count},
        "expected_test_fraud_approx": 98,
    }
    (PROCESSED_DATA_DIR / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Preprocessing complete.")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Fraud count in test set:", test_fraud_count)


if __name__ == "__main__":
    main()
