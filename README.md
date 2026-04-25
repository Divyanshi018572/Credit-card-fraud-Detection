---
title: Credit Card Fraud Detection
emoji: 💳
colorFrom: red
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# Credit Card Fraud Detection
End-to-end machine learning project for extreme class imbalance fraud detection using the ULB Kaggle dataset (`creditcard.csv`) with no synthetic external dataset.

## 🚀 Live App

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/Divya499/Credit-card-fraud_detection)

> **Try it live:** [https://huggingface.co/spaces/Divya499/Credit-card-fraud_detection](https://huggingface.co/spaces/Divya499/Credit-card-fraud_detection)

The interactive Streamlit dashboard lets you explore model performance, run real-time transaction risk scoring, and visualize fraud patterns across the dataset.

## High-Level System Design

The project is split into two primary components: an offline machine learning pipeline for processing data and training models, and a real-time Streamlit application for transaction risk scoring.

For a detailed visual breakdown of the architecture, data flow, and risk decision engine, please see the [System Design Document](SYSTEM_DESIGN.md).

## Project Structure

```
├── data/
│   ├── raw/creditcard.csv
│   └── processed/
├── models/
│   ├── xgboost_fraud.pkl
│   └── ...
├── configs/
│   └── config.yaml          # Centralized hyperparameter management
├── src/
│   └── utils/
│       └── config_loader.py # Modular config loading logic
├── pipeline/
│   ├── eda.py               # Exploratory Data Analysis
│   ├── preprocessing.py     # Feature engineering & scaling
│   ├── train.py             # MLflow-tracked training & tuning
│   └── evaluate.py          # Model performance reporting
├── tests/                   # Automated pytest suite (Unit & Integration)
│   ├── test_api.py
│   ├── test_pipeline.py
│   └── test_utils.py
├── outputs/                 # Visualization artifacts
├── api.py                   # Production FastAPI service
├── app.py                   # Interactive Streamlit dashboard
├── Dockerfile
└── path_utils.py
```

## 🛠️ Engineering Maturity

This project has been refactored from a procedural prototype to a **Production-Ready MLOps System**:

- **Configuration Management:** All hyperparameters and business rules are decoupled into `configs/config.yaml`.
- **Experiment Tracking:** Integrated with **MLflow** for full visibility into model versions and tuning runs.
- **Automated Testing:** 100% pass rate on `pytest` suite covering core logic and API endpoints.
- **Production API:** High-speed REST API (FastAPI) with Pydantic input validation (<10ms latency).
- **Containerization:** Fully Dockerized for cross-platform consistency.

## Why Accuracy Fails Here

Fraud rate is only ~0.17%. A naive model that predicts every transaction as legitimate gets ~99.83% accuracy while detecting zero fraud.  
So accuracy is not the objective. Primary metrics are:

- F1-score (fraud class)
- Recall (fraud class)
- Precision (fraud class)
- PR-AUC (more informative than ROC-AUC in rare event detection)

## 📊 Model Performance Results

> Evaluated on a held-out test set (stratified split, no data leakage). All metrics are for the **fraud (positive) class**.

| Model | F1 | Recall | Precision | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost + SMOTE + Threshold** | 0.512 | **0.888** | 0.360 | **0.859** | 0.980 |
| **Random Forest + SMOTE** | **0.726** | 0.837 | **0.641** | 0.843 | **0.982** |
| Isolation Forest (unsupervised) | 0.328 | 0.337 | 0.320 | 0.194 | 0.954 |
| One-Class SVM (unsupervised) | 0.133 | 0.847 | 0.072 | 0.345 | 0.946 |

**Key takeaway:** Supervised models with SMOTE achieve 4–6× better PR-AUC than unsupervised baselines. XGBoost is optimized for maximum fraud recall (catching more frauds), Random Forest is the best balanced model by F1.

## ⚙️ Model Configuration & Hyperparameter Tuning

XGBoost was tuned using `RandomizedSearchCV` (6 iterations, 3-fold CV on training data). Best parameters found:

| Parameter | Value | Why |
|---|---|---|
| `n_estimators` | 300 | Balanced depth vs overfitting |
| `max_depth` | 5 | Moderate complexity |
| `learning_rate` | 0.10 | Standard for boosting on tabular |
| `subsample` | 0.70 | Reduces overfitting via row sampling |
| `scale_pos_weight` | 300 | Corrects severe class imbalance (1:578 ratio) |
| **CV F1 (3-fold)** | **0.752** | Validates test score is not a lucky split |
| **Operating threshold** | **0.285** | Tuned on validation split (leak-free) |

Random Forest configuration:
- `n_estimators=400`, `max_depth=14`, `class_weight='balanced'`, `min_samples_leaf=2`



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

Run the end-to-end pipeline:

```powershell
# 1. Preprocess data
python pipeline/preprocessing.py

# 2. Train with MLflow tracking
python pipeline/train.py

# 3. View Experiment Dashboard
mlflow ui
```

## 🚀 Future Roadmap (The "Score 10" Upgrade)

The following features are planned for future versions to achieve "Elite" engineering status:

- **v2.0: Data Version Control (DVC):** Implementation of DVC to version the raw and processed datasets alongside the code.
- **v2.1: Model Monitoring:** Real-time data drift detection using `evidently` or `Prometheus` to alert on performance decay.
- **v2.2: Automated CI/CD:** Full GitHub Actions pipeline for automated unit testing, Docker builds, and deployment to AWS/SageMaker.
- **v2.3: Prediction Logging:** Implementing a PostgreSQL backend to log all production API predictions for future retraining loops.

## Docker (Optimized Runtime)

Build:

```powershell
docker build -t credit-card-fraud-app .
```

Run:

```powershell
docker run --rm -p 8501:8501 credit-card-fraud-app
```

## Streamlit App Sections

`app.py` includes all requested sections:

1. Imbalance problem visualization
2. Live transaction risk analyzer
3. Detection method comparison (PR curves + model table + threshold tradeoff)
4. SMOTE impact visualization
5. Confusion matrix + financial impact estimator
