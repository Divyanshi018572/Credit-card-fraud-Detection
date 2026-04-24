# notes.md — ML + Product Thinking + Interview Mastery
> Credit Card Fraud Detection | ULB Kaggle Dataset | End-to-End ML Pipeline

---

## Section 1 — Business Problem Framing

**Real-world problem:** Automatically detect fraudulent credit card transactions in real time to prevent financial loss to cardholders and issuing banks.

**Stakeholders:**
- Cardholders (don't want unauthorized charges)
- Issuing banks (absorb fraud losses, pay chargebacks)
- Merchants (chargeback disputes damage revenue)
- Risk/Fraud Analytics teams (own the model)
- Card networks (Visa/Mastercard SLA compliance)

**Business KPI this improves:**
- Fraud loss rate (EUR saved per period)
- False Positive Rate (legitimate transactions blocked — damages customer experience)
- Detection Rate / Recall on fraud class
- Chargeback ratio (regulatory concern above ~1%)

**Why ML over rules?**
Rules (`if Amount > 5000 AND country != home_country`) are brittle, easily gamed by fraudsters, and produce massive false positive rates. ML learns non-linear interaction patterns across 29 features simultaneously — patterns no human analyst can hand-code.

**What if model fails silently in production?**
- Fraud slips through undetected → direct financial loss
- Legitimate transactions blocked → customer churn, brand damage
- Model output distribution shifts → metrics look fine but fraud rate climbs
- Silent failure is THE most dangerous production scenario — requires active monitoring

---

## Section 2 — Dataset Intelligence

- **Dataset:** ULB Credit Card Fraud Dataset (Kaggle)
- **Scale:** 284,807 transactions, 31 columns, 48-hour window (Sept 2013, European cardholders)
- **Feature types:** All numerical — V1–V28 are PCA-transformed (privacy), Amount, Time, Class
- **Label distribution:** 284,315 legitimate (99.83%) vs 492 fraud (0.17%) — extreme imbalance
- **Missing values:** Zero (confirmed in `01_eda.py` — `df.isna().sum().sum() == 0`)
- **Sparsity:** None — all values populated

**Data leakage risks:**
1. Applying SMOTE before train/test split → synthetic fraud in test set (AVOIDED — split done first)
2. Fitting StandardScaler on full dataset then splitting → test data influences scaler (AVOIDED — scaler fit on train only)
3. Threshold tuning on test set → optimistic threshold estimate (AVOIDED — threshold tuned on internal validation split)
4. Using `Time` feature without bucketing → could encode positional leakage in time-ordered data (PRESENT — Time is dropped, not used)

**Bias risks:**
- V1–V28 are anonymized — demographic features could be encoded inside PCA components
- European cardholder population only — model may not generalize to other geographies
- 48-hour window — seasonal and behavioral drift not captured

---

## Section 3 — Data Preprocessing Deep-Dive

### Step 1: Log-transform Amount
- **What:** `df["log_Amount"] = np.log1p(df["Amount"])`
- **Why:** Amount is right-skewed (most transactions small, few very large). Log transform compresses the range, stabilizing gradient-based models.
- **Alternative:** RobustScaler directly on Amount; Box-Cox transform
- **Trade-off:** Gains: model convergence, reduced outlier impact. Loses: interpretability of original scale
- **Production concern:** If fraud shifts to micro-transactions (< €1), log compression may mask the signal

### Step 2: Drop Amount and Time
- **What:** `df = df.drop(columns=["Amount", "Time"])`
- **Why:** Raw Amount replaced by log_Amount. Time is absolute seconds — not a useful feature without engineering (hour-of-day, day-of-week)
- **Alternative:** Extract Time features (hour, day bucket, velocity features)
- **Trade-off:** Simplicity vs information loss. Time-based features could catch velocity fraud (10 transactions in 60 seconds)
- **Production concern:** Velocity patterns (key fraud signal) are completely ignored

### Step 3: Stratified Train/Test Split (80/20)
- **What:** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`
- **Why:** Stratify preserves 0.17% fraud rate in both splits. Without it, test set could have 0 fraud samples.
- **Alternative:** Time-based split (transactions before date X = train, after = test)
- **Trade-off:** Random stratified is reproducible and unbiased. Time-based is more realistic but harder to implement.
- **Production concern:** Random split assumes i.i.d. data. Real fraud patterns drift over time — time-based split would be more honest.

### Step 4: StandardScaler on log_Amount Only
- **What:** `scaler.fit_transform(X_train[["log_Amount"]])` then `transform` on test
- **Why:** V1–V28 are already PCA-normalized. Only log_Amount needs scaling.
- **Alternative:** Scale all features; use RobustScaler (handles outliers better)
- **Trade-off:** Correct choice — over-scaling PCA features would distort their pre-existing normalization
- **Production concern:** Scaler parameters (mean, std) must be saved and reloaded for inference — correctly done with `joblib.dump(scaler, ...)`

### Step 5: SMOTE (supervised models only)
- **What:** `SMOTE(sampling_strategy=0.1)` applied on training data only
- **Why:** Generates synthetic minority (fraud) samples to bring fraud ratio to 10% of training set
- **Alternative:** Undersampling legitimate class; `class_weight='balanced'`; ADASYN
- **Trade-off:** SMOTE improves minority learning but can generate unrealistic samples in sparse regions
- **Production concern:** SMOTE is a training-only artifact. Must not be applied at inference. Correctly implemented.

---

## Section 4 — Model Selection (Bar Raiser Level)

### Model 1: Isolation Forest
- **Why chosen:** Unsupervised baseline — detects anomalies without labels by isolating points using random splits
- **Math intuition:** Fraudulent transactions are rare and different → require fewer random splits to isolate → shorter path length = higher anomaly score
- **Key hyperparameters:** `n_estimators=200` (more trees = more stable), `contamination=0.0017` (matches known fraud rate)
- **Training vs inference:** Fast training O(n log n), fast inference O(log n) per sample
- **Failure modes:** Fails when fraud is distributed similarly to legitimate (low-value fraud); sensitive to contamination hyperparameter calibration
- **Alternatives:** Local Outlier Factor (density-based), Autoencoder (learns reconstruction error), DBSCAN
- **Interpretability:** Low — no feature importance without SHAP. Business impact: hard to explain flagged transactions to compliance teams
- **Production complexity:** Simple — single model, no preprocessing dependency

### Model 2: One-Class SVM
- **Why chosen:** Learns boundary around "normal" behavior using only legitimate transactions
- **Math intuition:** Finds a hypersphere (or hyperplane in kernel space) containing most legitimate samples; points outside = anomalies
- **Key hyperparameters:** `kernel='rbf'`, `nu=fraud_rate` (controls fraction of outliers), `gamma='scale'`
- **Failure modes:** Extremely slow on large datasets (O(n²) kernel computation) — subsampled to 50K legitimate samples. Very sensitive to nu parameter.
- **Alternatives:** Isolation Forest (faster), Deep SVDD (neural version)
- **Interpretability:** Very low — kernel space decisions uninterpretable
- **Production complexity:** Complex — slow inference, requires careful subsampling strategy

### Model 3: Random Forest + SMOTE
- **Why chosen:** Strong supervised ensemble; handles non-linear interactions; `class_weight='balanced'` provides additional imbalance correction on top of SMOTE
- **Math intuition:** Builds 400 decision trees on bootstrap samples with random feature subsets; aggregates probability estimates
- **Key hyperparameters:** `n_estimators=400`, `max_depth=14`, `min_samples_leaf=2`, `class_weight='balanced'`
- **Best result:** F1=0.726, PR-AUC=0.843
- **Failure modes:** No probability calibration → raw probabilities may be overconfident; high memory footprint (400 deep trees × 30 features)
- **Alternatives:** LightGBM (faster, similar performance), CatBoost (handles categoricals natively), Gradient Boosting
- **Interpretability:** Medium — feature importances available, SHAP values computable
- **Production complexity:** Moderate — large pickle file (~33MB), memory-intensive inference

### Model 4: XGBoost + SMOTE + Threshold Tuning
- **Why chosen:** Best PR-AUC (0.859); gradient boosting is state-of-art for tabular imbalanced data; explicit scale_pos_weight handles imbalance; custom threshold maximizes fraud recall
- **Math intuition:** Sequentially adds trees that correct residual errors of previous ensemble; `scale_pos_weight=300` upweights fraud errors 300×
- **Best result:** F1=0.512 (at threshold=0.285), Recall=0.888, PR-AUC=0.859
- **Key hyperparameters (from RandomizedSearchCV):** n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.7, scale_pos_weight=300
- **CV F1 (3-fold on training):** 0.752
- **Failure modes:** Default threshold 0.5 gives poor recall; overconfident on seen fraud patterns
- **Alternatives:** LightGBM (faster training), TabNet (attention-based tabular), Neural network with focal loss
- **Interpretability:** Medium (SHAP available for XGBoost natively)
- **Production complexity:** Moderate — ImbPipeline with SMOTE step baked in

---

## Section 5 — Training Strategy Audit

- **Split strategy:** Stratified 80/20 random split ✅ (correct for this domain — preserves class ratio)
- **Validation methodology:** Hold-out + internal 80/20 within training for threshold tuning ✅. XGBoost uses 3-fold CV via RandomizedSearchCV ✅
- **Hyperparameter tuning:** RandomizedSearchCV with 6 iterations, 3-fold CV for XGBoost ✅. Random Forest uses manually selected params ⚠️ (not tuned)
- **Overfitting signals:** No learning curves plotted. No train vs validation metric comparison logged. ❗
- **Regularization:** XGBoost: subsample=0.7 (row sampling), max_depth=5 (tree depth limit). RF: min_samples_leaf=2, max_depth=14. Class weights as soft regularization.

**What a production training pipeline SHOULD look like:**
1. StratifiedKFold (5-fold) for all models — not just XGBoost
2. MLflow / W&B experiment tracking with all hyperparameter combinations logged
3. Model versioning with DVC or MLflow Model Registry
4. Automated retraining trigger when PSI (Population Stability Index) > 0.2
5. Calibrated probabilities using `CalibratedClassifierCV`

---

## Section 6 — Evaluation Strategy Audit

| Metric | Business meaning | When misleading | What to add |
|---|---|---|---|
| F1 (fraud) | Harmonic mean of precision/recall for fraud | Hides threshold sensitivity | Calibration curve |
| Recall (fraud) | % of real frauds caught | Ignores false alarm cost | Expected fraud $ saved |
| Precision (fraud) | % of flagged = real fraud | High precision ≠ catching all fraud | False positive rate per 1K transactions |
| PR-AUC | Area under P-R curve — best for rare class | N/A — correctly chosen | ✅ Correct primary metric |
| ROC-AUC | Discriminative power | Inflated by TN (99.83% legit) — misleading here | Use only as secondary |

**Ideal evaluation suite for fraud detection:**
1. PR-AUC (primary — implemented ✅)
2. Recall@Precision=0.9 (business threshold: at 90% precision, how many frauds do we catch?)
3. Expected $ saved = TP × avg_fraud_amount - FP × review_cost
4. Detection latency (P99 inference time)
5. Calibration curve (are probabilities trustworthy?)
6. KS statistic (fraud score distribution separation)

---

## Section 7 — Production Readiness Scorecard

| Dimension | Score | Reason |
|---|---|---|
| Data pipeline robustness | 3/5 | Modular scripts, but no schema validation, no retry logic |
| Model training quality | 4/5 | Stratified split, SMOTE, threshold tuning, CV for XGBoost |
| Evaluation rigor | 4/5 | PR-AUC, F1, Recall, Precision — correct metrics. No calibration curve. |
| Inference pipeline | 3/5 | Streamlit app works but no REST API, no input validation schema |
| API layer | 1/5 | No REST API — Streamlit UI only |
| Error handling | 2/5 | Basic FileNotFoundError in EDA; no input validation in app |
| Monitoring | 1/5 | No drift detection, no logging, no alerting |
| Scalability | 2/5 | Single-instance Streamlit, no batching, no caching |
| Reproducibility | 4/5 | random_state=42 throughout, joblib serialization, requirements.txt |
| Documentation | 4/5 | README, SYSTEM_DESIGN.md, inline comments |

**Overall: 28/50 — Strong Mid-level signal for a fresher. Junior→Mid transition.**

---

## Section 8 — End-to-End Pipeline (Exact)

```
RAW DATA (creditcard.csv — 284,807 rows × 31 cols)
   ↓ [01_eda.py: compute class distribution, feature correlations, generate 6 plots, save feature_ranges.json]
   ↓ [02_preprocessing.py: log1p(Amount), drop Amount+Time, stratified 80/20 split, StandardScaler on log_Amount only, save train.csv + test.csv + scaler]
PROCESSED FEATURES (train: 227,845 rows × 29 cols | test: 56,962 rows × 29 cols)
   ↓ [03_train.py — 4 parallel training tracks]
     Track A: IsolationForest(n_est=200, contamination=0.0017) → fit on X_train
     Track B: OneClassSVM(rbf, nu=0.0017) → fit on X_train[y_train==0].sample(50K)
     Track C: SMOTE(0.1) → RandomForest(400, depth=14) → fit on SMOTE-augmented train
     Track D: 80/20 internal split → RandomizedSearchCV(XGBPipeline, 6 iter, 3-fold) → threshold sweep on val → refit best params on full train
   ↓ [Serialize: 4 .pkl models + scaler via joblib, model_metadata.json with best params + threshold]
TRAINED ARTIFACTS (models/*.pkl)
   ↓ [04_evaluate.py: load all models, score test set, compute PR-AUC/F1/Recall/Precision/ROC-AUC, plot PR curves + confusion matrix + threshold curve, SMOTE impact visualization]
EVALUATION OUTPUTS (outputs/*.png, *.csv, *.json)
   ↓ [app.py: load all artifacts at startup (@st.cache_data), render 5-section Streamlit dashboard]
FINAL OUTPUT: Interactive fraud risk dashboard + real-time transaction scorer
```

---

## Section 9 — STAR Method (Interview Ready)

**Situation:** European banks face 0.17% credit card fraud rate — a severe class imbalance making standard ML approaches ineffective. Existing rule-based systems miss adaptive fraud patterns.

**Task:** Build a production-grade end-to-end ML pipeline that detects fraud with high recall (minimize missed frauds) while controlling false positives (minimize legitimate transaction blocks).

**Action:** Designed a 4-stage modular pipeline. In preprocessing, engineered log-transformed Amount feature and applied Stratified 80/20 split with StandardScaler fitted exclusively on training data to prevent leakage. Implemented four models — two unsupervised baselines (Isolation Forest, One-Class SVM) and two supervised models with SMOTE augmentation (Random Forest, XGBoost). For XGBoost, applied RandomizedSearchCV with 3-fold CV across learning rate, tree depth, subsample, and scale_pos_weight — then tuned the decision threshold on a held-out validation split (not the test set) to optimize for fraud recall in the 0.2–0.3 probability range. Selected PR-AUC as the primary metric over ROC-AUC because ROC is inflated by the 99.83% legitimate majority class. Deployed as a live Streamlit dashboard on Hugging Face Spaces with automated GitHub Actions CI/CD pipeline.

**Result:** XGBoost achieved PR-AUC of 0.859 and Recall of 88.7% at threshold 0.285 — catching ~9 in 10 fraud cases. Random Forest achieved best balanced F1 of 0.726. Supervised models outperformed unsupervised baselines by 4–6× on PR-AUC. Live app deployed at huggingface.co/spaces/Divya499/Credit-card-fraud_detection with full CI/CD automation.

---

## Section 10 — Multi-Level Interview Explanation

### 🔹 30-Second (HR / Non-technical)
"I built a system that automatically detects fraudulent credit card transactions. Because fraud is extremely rare — only 1 in 600 transactions — I used advanced techniques to teach the model to catch fraud without flagging too many legitimate purchases. The system is live online and can analyze any transaction in real time."

### 🔹 2-Minute (Recruiter / ML screening)
"The project tackles credit card fraud detection on the ULB Kaggle dataset with a 0.17% fraud rate — extreme class imbalance. I built a 4-stage pipeline: EDA, preprocessing, training, and evaluation. For preprocessing I applied log-transform on transaction amounts and stratified train/test splits. I trained four models — Isolation Forest and One-Class SVM as unsupervised baselines, plus Random Forest and XGBoost with SMOTE oversampling. I used PR-AUC as the primary metric because ROC-AUC is misleading on imbalanced data. Best result: XGBoost with threshold tuning achieved 88.7% recall and 0.859 PR-AUC. The whole thing is deployed on Hugging Face with GitHub Actions CI/CD."

### 🔹 5-Minute (ML Engineer deep-dive)
"The core challenge was class imbalance — 284K legitimate vs 492 fraud. Standard accuracy is useless here (99.83% by predicting all legitimate). I chose PR-AUC as the primary metric. For preprocessing: log1p transform on Amount (right-skewed distribution), drop raw Time (would need velocity engineering to be useful), StandardScaler on log_Amount only (V1-V28 are already PCA-normalized). SMOTE with sampling_strategy=0.1 applied only on training data — post-split — to avoid test contamination. For XGBoost: RandomizedSearchCV with 6 iterations, 3-fold CV, searched over n_estimators, max_depth, learning_rate, subsample, and scale_pos_weight (300–700 range to correct 1:578 imbalance). Threshold was tuned on a separate internal validation split — NOT the test set — to avoid optimistic threshold estimates. Final threshold: 0.285 giving Recall=0.888, Precision=0.360. This Recall/Precision trade-off is intentional: in fraud detection catching fraud is more important than false alarms. The app is deployed on Hugging Face with Streamlit, automated via GitHub Actions."

### 🔹 10-Minute (System Design + Bar Raiser)
"Let me walk you through the full architecture and the trade-offs I made. The pipeline has four discrete stages: EDA generates distribution plots and feature correlation analysis which confirmed V14, V10, V12 as top fraud-correlated features (negative correlation = lower values indicate fraud). Preprocessing drops raw Time — a controversial decision: Time could encode velocity features (number of transactions per hour) which are strong fraud signals, but that would require sequential processing not compatible with this batch approach. I log-transform Amount and scale it with StandardScaler fitted only on train — a critical leakage prevention step. The 80/20 stratified split preserves the 0.17% fraud rate in both partitions. Training runs four models. Isolation Forest is the unsupervised baseline — uses contamination=0.0017 matching the known fraud rate. One-Class SVM is subsampled to 50K legitimate samples because OCSVM is O(n²) in training. For supervised models, SMOTE brings fraud to 10% of training set before Random Forest. XGBoost uses an internal 80/20 split for threshold tuning — this is a leak-prevention technique: threshold tuned on validation, evaluated on held-out test. scale_pos_weight=300 was found by RandomizedSearchCV to be the optimal imbalance correction weight. For deployment, I containerized with Docker and deployed via Hugging Face Spaces. The GitHub Action uses the huggingface_hub Python API (not the CLI — CLI had Unicode encoding issues on Ubuntu runners) to sync code files on every push. If I were to scale this to production: I'd add a FastAPI layer with Pydantic input validation, Redis caching for repeated transactions, Kafka for streaming transaction events, Prometheus/Grafana for model monitoring, and PSI-based drift detection triggering automatic retraining."

---

## Section 11 — Bar Raiser Interview Questions (30 Questions)

### ML Theory (5 questions):
1. Why is ROC-AUC misleading for fraud detection but PR-AUC is appropriate? Walk me through the math.
2. What exactly does SMOTE generate — explain the interpolation algorithm. What are its failure modes in high-dimensional spaces?
3. XGBoost uses `scale_pos_weight` for imbalance. How is this different from SMOTE? Can you use both simultaneously?
4. What is the difference between threshold tuning and probability calibration? Why might you need both?
5. Isolation Forest isolates anomalies through random splits. What is its theoretical failure mode on datasets where fraud clusters in specific feature subspaces?

### This Specific Project (10 questions):
6. Why did you drop the `Time` feature? What fraud signal does this lose?
7. Your XGBoost threshold is 0.285. If a bank says "we can only review 50 flagged transactions per day," how would you adjust this?
8. SMOTE is applied with `sampling_strategy=0.1`. What happens if you use 0.5? Show this in your evaluation.
9. You used `RandomizedSearchCV` with only 6 iterations. Why not GridSearchCV? What did you miss?
10. The One-Class SVM was subsampled to 50K rows. How does this affect the learned decision boundary?
11. Your log_Amount scaler is saved as `log_amount_scaler.pkl`. What happens at inference if a transaction has Amount=-1?
12. The confusion matrix shows 7 false negatives at your threshold. What is the expected financial impact in EUR?
13. How would you detect if fraud patterns shift 3 months after deployment without access to new labels?
14. Your Random Forest uses `class_weight='balanced'` AND SMOTE. Is this double-counting the imbalance correction?
15. V14 has the highest negative correlation with fraud. What does this mean operationally?

### Edge Cases & Failure Modes (5 questions):
16. What happens if the user enters Amount=0 in the live demo? Does the model handle this correctly?
17. The Isolation Forest's anomaly score is min-max normalized to [0,1]. What breaks if all test scores are the same value?
18. Your app loads all 4 models at startup. What happens on Hugging Face if the 33MB Random Forest pkl exceeds memory?
19. What if a fraudster deliberately sends transactions with V14=0 (the median value)? Will your model catch it?
20. The SMOTE strategy was evaluated for values [0.05, 0.1, 0.2, 0.5]. What if 0.1 is a local optimum?

### System Design Extensions (5 questions):
21. How would you serve this model to process 100K transactions per second with sub-50ms P99 latency?
22. Design a retraining trigger: what signals indicate the model needs retraining, and how would you automate it?
23. How would you A/B test XGBoost vs Random Forest in production without exposing users to fraud risk?
24. What database schema would you use to store transaction predictions for audit trail and compliance?
25. How would you handle the case where a new fraud pattern emerges that none of your 4 models have seen?

### Trap Questions (5 questions):
26. "Your model has 99.83% accuracy — that's excellent, right?"  *(Trap: Accuracy is meaningless here)*
27. "You used SMOTE, so now your training set has balanced classes — does that mean you should use 0.5 as threshold?" *(Trap: SMOTE on training doesn't change test distribution)*
28. "The ROC-AUC is 0.98 — your model is almost perfect." *(Trap: ROC-AUC is inflated by 284K true negatives)*
29. "Since V1-V28 are already PCA features, you don't need StandardScaler." *(Trap: log_Amount is NOT PCA-normalized and still needs scaling)*
30. "Isolation Forest has 0.95 ROC-AUC — it's almost as good as XGBoost." *(Trap: PR-AUC is 0.19 vs 0.86 — ROC-AUC hides the performance gap)*

---

## Section 12 — Honest Senior-Level Critique

### What signals JUNIOR-level thinking:
- `requirements.txt` has no version pins (`pandas`, `numpy` etc.) — will break in 6 months when APIs change
- `Time` feature dropped instead of engineered — velocity features are the #1 signal in real fraud detection
- No input validation in Streamlit app — `st.slider` can produce values outside training distribution silently
- Upload helper scripts (`upload_script.py`, `upload_commit.py`) committed to the repo — these are developer artifacts, not production code
- `random_state=42` everywhere is good, but there's no seed management system or config file
- Scaler applied only to `log_Amount` — correct, but not documented with a comment explaining WHY V1-V28 are excluded

### What is MISSING that any production system must have:
- REST API (FastAPI/Flask) with Pydantic input validation
- Structured JSON logging (no logs anywhere in the codebase)
- Model monitoring / drift detection (PSI, KS test)
- Experiment tracking (MLflow, W&B, or even a simple metrics CSV with timestamps)
- Probability calibration (`CalibratedClassifierCV`)
- Unit tests for any function
- Health check endpoint
- Pinned dependency versions
- `.env` / secrets management (token hardcoded in upload scripts)
- Model registry / versioning strategy
- Retraining pipeline trigger

### What would make a FAANG hiring manager skip this resume:
- If presented as "99.83% accuracy" — instant reject
- No API means no production deployment experience signal
- No monitoring means the candidate doesn't think in production terms
- No tests of any kind

### What is ACTUALLY good about this project:
- Correct primary metric (PR-AUC over ROC-AUC) — shows real domain understanding
- Stratified split with SMOTE post-split — leakage prevention done correctly
- Threshold tuning on validation split, not test set — technically sound
- 4 models (2 unsupervised + 2 supervised) with principled comparison — shows breadth
- Live deployed app + GitHub Actions CI/CD — very rare for a fresher portfolio
- Modular 4-script pipeline instead of a single notebook — production-thinking signal
- `class_weight='balanced'` + SMOTE together is valid (they operate at different levels)

---

## Section 13 — FAANG Upgrade Roadmap

### Quick wins (1–2 days):
- Pin all `requirements.txt` versions (`pandas==2.2.2`, `xgboost==2.0.3`, etc.)
- Add `.env` file for HF token — remove hardcoded token from upload scripts
- Delete `upload_script.py` and `upload_commit.py` from the repo (developer artifacts)
- Add `# type: ignore` comments → replace with actual type hints throughout
- Add a `CONTRIBUTING.md` and `LICENSE` file

### Medium effort (1–2 weeks):
- Add FastAPI endpoint: `POST /predict` with Pydantic schema validation
- Add velocity features from `Time`: transactions-per-hour, time-since-last-transaction
- Add probability calibration: `CalibratedClassifierCV(xgb_model, method='isotonic')`
- Add 10 unit tests: `test_preprocessing.py`, `test_inference.py`, `test_metrics.py`
- Add structured logging with Python `logging` module (JSON format)
- Add MLflow tracking for all training runs

### Production-grade (1 month+):
- Replace Streamlit with FastAPI backend + React/Next.js frontend
- Implement PSI-based drift detection with automated retraining trigger
- Add Redis caching for repeated transaction hashes
- Add SHAP explainability layer for each prediction
- Add Prometheus metrics + Grafana dashboard
- Implement time-based train/test split to reflect real deployment conditions
- Add StratifiedKFold (5-fold) cross-validation for all 4 models

---

## Section 14 — Hiring Signal Summary

| Signal | Assessment |
|---|---|
| Role Fit | ML Engineer / Data Scientist (tabular, imbalanced, production-aware) |
| Seniority Signal | Strong Junior → entry Mid-level |
| Resume Impact | Medium-Strong (deployment + pipeline elevate it above average fresher projects) |
| Interview Talking Points | 1) PR-AUC vs ROC-AUC choice 2) Leakage-safe threshold tuning 3) Live CI/CD deployment |
| Hiring Strength Score | 7/10 — above average fresher, below production-experienced candidate |
