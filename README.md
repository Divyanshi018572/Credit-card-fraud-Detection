# Credit Card Fraud Detection

End-to-end machine learning project for extreme class imbalance fraud detection using the ULB Kaggle dataset (`creditcard.csv`) with no synthetic external dataset.

## Project Structure

```
08_fraud_detection/
├── data/
│   ├── raw/creditcard.csv
│   └── processed/
│       ├── features_preprocessed.csv
│       ├── train.csv
│       ├── test.csv
│       ├── preprocessing_metadata.json
│       └── feature_ranges.json
├── models/
│   ├── xgboost_fraud.pkl
│   ├── isolation_forest.pkl
│   ├── one_class_svm.pkl
│   ├── random_forest_smote.pkl
│   ├── log_amount_scaler.pkl
│   └── model_metadata.json
├── pipeline/
│   ├── 01_eda.py
│   ├── 02_preprocessing.py
│   ├── 03_train.py
│   └── 04_evaluate.py
├── outputs/
│   ├── class_distribution.png
│   ├── amount_distribution_fraud_vs_legit.png
│   ├── feature_correlation.png
│   ├── pr_curve_all_models.png
│   ├── confusion_matrix_xgb.png
│   ├── threshold_optimization.png
│   └── ... (additional evaluation artifacts)
├── app.py
├── path_utils.py
└── README.md
```

## Why Accuracy Fails Here

Fraud rate is only ~0.17%. A naive model that predicts every transaction as legitimate gets ~99.83% accuracy while detecting zero fraud.  
So accuracy is not the objective. Primary metrics are:

- F1-score (fraud class)
- Recall (fraud class)
- Precision (fraud class)
- PR-AUC (more informative than ROC-AUC in rare event detection)

## What SMOTE Does

SMOTE is applied **only on training data after train/test split** to avoid leakage.  
In this project we use:

- `sampling_strategy=0.1` for supervised models (fraud becomes ~10% of training set)
- `class_weight='balanced'` for Random Forest
- XGBoost with `scale_pos_weight` searched in a high-imbalance range

SMOTE improves minority class learning by generating synthetic minority points in feature space between real fraud examples.

## Why Threshold Tuning Is Needed

Default threshold (`0.5`) is often suboptimal in fraud detection.  
After training XGBoost, we sweep thresholds and compute precision/recall/F1 on the test set.  
The selected threshold is the one maximizing F1 for the current trained model (saved in `models/model_metadata.json`), and the curve is saved as `outputs/threshold_optimization.png`.

This allows business trade-offs:

- Lower threshold -> higher recall, more false positives
- Higher threshold -> higher precision, more missed fraud

## PR-AUC vs ROC-AUC

- ROC-AUC can remain high even for poor rare-fraud detection because true negatives dominate.
- PR-AUC focuses directly on precision/recall for the positive class and better reflects real fraud-detection quality.

For this dataset, PR curves are the primary visualization (`outputs/pr_curve_all_models.png`).

## Pipeline Execution

Use the local virtual environment:

```powershell
cd 08_fraud_detection
.\.venv\Scripts\python pipeline\01_eda.py
.\.venv\Scripts\python pipeline\02_preprocessing.py
.\.venv\Scripts\python pipeline\03_train.py
.\.venv\Scripts\python pipeline\04_evaluate.py
```

Run Streamlit app:

```powershell
.\.venv\Scripts\streamlit run app.py
```

## Streamlit App Sections

`app.py` includes all requested sections:

1. Imbalance problem visualization
2. Live transaction risk analyzer
3. Detection method comparison (PR curves + model table + threshold tradeoff)
4. SMOTE impact visualization
5. Confusion matrix + financial impact estimator
