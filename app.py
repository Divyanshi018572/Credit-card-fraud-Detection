import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix

from path_utils import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR
from src.utils.config_loader import load_config

# Load global configuration
config = load_config()

st.set_page_config(
...
@st.cache_data
def load_data() -> dict:
    """
    Loads all precomputed data artifacts for the dashboard.
    
    Returns:
        dict: Collection of dataframes and metadata.
    """
...
@st.cache_resource
def load_models() -> dict:
    """
    Loads serialized model and scaler objects.
    
    Returns:
        dict: Collection of joblib-loaded models.
    """
...
def risk_tier(probability: float) -> str:
    """
    Maps a fraud probability to a business risk tier based on config thresholds.
    
    Args:
        probability (float): Raw model output probability.
        
    Returns:
        str: Risk category (Critical, High, Medium, Low).
    """
    tiers = config["business_logic"]["risk_tiers"]
    if probability >= tiers["critical"]:
        return "Critical"
    if probability >= tiers["high"]:
        return "High"
    if probability >= tiers["medium"]:
        return "Medium"
    return "Low"


def decision_action(tier: str) -> str:
    """
    Recommends a business action based on the risk tier.
    
    Args:
        tier (str): Risk category.
        
    Returns:
        str: Recommended action (Block, Flag, Approve).
    """
    if tier in {"Critical", "High"}:
        return "Block Transaction"
    if tier == "Medium":
        return "Flag for Review"
    return "Approve"


MODEL_INFO = {
    "Isolation Forest": {
        "family": "Unsupervised anomaly",
        "description": "Fast anomaly baseline using isolation paths.",
    },
    "One-Class SVM": {
        "family": "Unsupervised anomaly",
        "description": "Boundary model for normal-only behavior.",
    },
    "Random Forest + SMOTE": {
        "family": "Supervised ensemble",
        "description": "Balanced tree ensemble with synthetic minority samples.",
    },
    "XGBoost + SMOTE + Threshold": {
        "family": "Supervised boosting",
        "description": "Boosted trees with imbalance handling and tuned decision threshold.",
    },
}


def extract_feature_importances(model_obj) -> np.ndarray | None:
    if hasattr(model_obj, "feature_importances_"):
        return np.asarray(model_obj.feature_importances_)
    if hasattr(model_obj, "named_steps") and "xgb" in model_obj.named_steps:
        xgb_step = model_obj.named_steps["xgb"]
        if hasattr(xgb_step, "feature_importances_"):
            return np.asarray(xgb_step.feature_importances_)
    return None


data = load_data()
models = load_models()
test_df = data["test_df"]

st.title("Credit Card Fraud Detection - Real-Time Transaction Risk Analyzer")

model_options = [
    "Isolation Forest",
    "One-Class SVM",
    "Random Forest + SMOTE",
    "XGBoost + SMOTE + Threshold",
]
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "XGBoost + SMOTE + Threshold"
if "threshold" not in st.session_state:
    st.session_state["threshold"] = float(np.clip(data["model_meta"]["xgb_best_threshold"], 0.1, 0.5))

st.sidebar.header("Model Controls")
# No `key` on the radio: return value is read directly each run.
# This means NO widget owns 'selected_model', so buttons can update it freely.
radio_choice = st.sidebar.radio(
    "Choose model",
    model_options,
    index=model_options.index(st.session_state["selected_model"]),
)
# Sync radio choice → session state (safe: no widget owns 'selected_model')
st.session_state["selected_model"] = radio_choice
selected_model = radio_choice

model_table = data["model_table"].copy()
model_table.index = model_table.index.astype(str)

st.header("Executive Overview")
eda = data["eda_summary"]
total_transactions = int(eda["rows"])
fraud_transactions = int(eda["class_distribution"]["fraud"])
fraud_rate_pct = float(eda["fraud_rate_percent"])
best_f1_model = str(model_table["F1"].idxmax())
best_f1_score = float(model_table.loc[best_f1_model, "F1"])
best_pr_model = str(model_table["PR_AUC"].idxmax())

overview_cols = st.columns(4)
overview_cols[0].metric("Total Transactions", f"{total_transactions:,}")
overview_cols[1].metric("Fraud Rate", f"{fraud_rate_pct:.3f}%", f"{fraud_transactions:,} fraud cases")
overview_cols[2].metric("Best F1 Model", best_f1_model, f"F1 {best_f1_score:.3f}")
overview_cols[3].metric("Operating Threshold", f"{st.session_state['threshold']:.2f}", "Adjust from sidebar")

st.markdown(
    (
        "This dashboard focuses on rare-event detection where accuracy is misleading. "
        f"Current strongest ranking model by PR-AUC is **{best_pr_model}**, "
        "and threshold tuning is used to balance fraud recall vs false alarms."
    )
)

card_cols = st.columns(4)
clicked_model = None  # track which card was tapped this run

for col, model_name in zip(card_cols, model_options):
    is_active = model_name == selected_model
    row = model_table.loc[model_name]
    card_class = "model-card active" if is_active else "model-card"
    col.markdown(
        (
            f"<div class='{card_class}'>"
            f"<div class='model-title'>{model_name}</div>"
            f"<div class='model-sub'>{MODEL_INFO[model_name]['family']}</div>"
            f"<div class='model-metrics'>F1: {row['F1']:.3f} | Recall: {row['Recall']:.3f}<br>"
            f"Precision: {row['Precision']:.3f}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    # use_container_width makes the button full-width under the card
    if col.button("Use this model", key=f"pick_{model_name}", use_container_width=True):
        clicked_model = model_name  # note which was tapped, don't rerun yet

# After ALL 4 cards are rendered, switch model and rerun once (single tap)
if clicked_model is not None:
    st.session_state["selected_model"] = clicked_model
    st.rerun()


default_threshold = float(np.clip(data["model_meta"]["xgb_best_threshold"], 0.1, 0.5))
threshold = st.sidebar.slider(
    "Threshold",
    min_value=0.1,
    max_value=0.5,
    value=float(st.session_state.get("threshold", default_threshold)),
    step=0.01,
    key="threshold",
)
show_smote_impact = st.sidebar.checkbox("SMOTE toggle (show impact)", value=True)

st.header("Section 1 - The Imbalance Problem")
col1, col2 = st.columns(2)
with col1:
    class_dist = data["eda_summary"]["class_distribution"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Legitimate", "Fraud"], [class_dist["legitimate"], class_dist["fraud"]], color=["#22c55e", "#ef4444"])
    ax.set_yscale("log")
    ax.set_title("Class Distribution (Log Scale)")
    ax.set_ylabel("Count (log scale)")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Accuracy is misleading here: predicting all legitimate gives ~99.83% accuracy while detecting zero fraud."
    )

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        data["naive_vs_smote"]["Model"],
        data["naive_vs_smote"]["F1"],
        color=["#64748b", "#22c55e"],
    )
    ax.set_ylim(0, 1)
    ax.set_title("Why Standard Models Fail (F1 Comparison)")
    ax.set_ylabel("F1 Score (Fraud)")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close(fig)

st.header("Section 2 - Transaction Risk Analyzer (Live Demo)")
selected_row = model_table.loc[selected_model]
st.markdown(
    (
        f"**Active Model:** `{selected_model}`  \n"
        f"{MODEL_INFO[selected_model]['description']}  \n"
        f"F1 `{selected_row['F1']:.3f}` | Recall `{selected_row['Recall']:.3f}` | "
        f"Precision `{selected_row['Precision']:.3f}`"
    )
)
top_features = ["V14", "V10", "V12", "V4", "V11"]
feature_ranges = data["feature_ranges"]
feature_columns = data["model_meta"]["feature_columns"]
feature_medians = test_df[feature_columns].median()
feature_means = test_df[feature_columns].mean()
feature_stds = test_df[feature_columns].std().replace(0, 1.0)

random_col, controls_col = st.columns([1, 2])
with random_col:
    sample_choice = st.radio("Randomize from real test set", ["None", "Random Legitimate", "Random Fraud"], index=0)
    if st.button("Load sample from test set"):
        subset = test_df.copy()
        if sample_choice == "Random Legitimate":
            subset = subset[subset["Class"] == 0]
        elif sample_choice == "Random Fraud":
            subset = subset[subset["Class"] == 1]
        if len(subset) > 0:
            st.session_state["sample_row"] = subset.sample(1, random_state=np.random.randint(0, 10_000)).iloc[0]

sample_row = st.session_state.get("sample_row")
input_vector = feature_medians.copy()
if sample_row is not None:
    for col in feature_columns:
        input_vector[col] = sample_row[col]

with controls_col:
    scaler = models["scaler"]
    default_amount = 100.0
    if sample_row is not None:
        scaled_log_amt = float(sample_row["log_Amount"])
        restored_log_amt = (scaled_log_amt * scaler.scale_[0]) + scaler.mean_[0]
        default_amount = float(np.expm1(restored_log_amt))

    amount = st.number_input("Transaction Amount (EUR)", min_value=0.0, max_value=5000.0, value=default_amount, step=1.0)
    for feat in top_features:
        f_min = float(feature_ranges[feat]["min"])
        f_max = float(feature_ranges[feat]["max"])
        default_val = float(input_vector[feat])
        default_val = float(np.clip(default_val, f_min, f_max))
        input_vector[feat] = st.slider(feat, min_value=f_min, max_value=f_max, value=default_val)

    scaled_log_amount = scaler.transform([[np.log1p(amount)]])[0][0]
    input_vector["log_Amount"] = scaled_log_amount

model_input = pd.DataFrame([input_vector[feature_columns]])
model_obj = models[selected_model]

if selected_model in {"Random Forest + SMOTE", "XGBoost + SMOTE + Threshold"}:
    fraud_probability = float(model_obj.predict_proba(model_input)[:, 1][0])
else:
    if selected_model == "Isolation Forest":
        raw_score = float(-model_obj.decision_function(model_input)[0])
        score_col = "IsolationForest_score"
    else:
        raw_score = float(-model_obj.decision_function(model_input)[0])
        score_col = "OneClassSVM_score"
    train_scores = data["score_df"][score_col]
    fraud_probability = float((raw_score - train_scores.min()) / (train_scores.max() - train_scores.min() + 1e-9))
    fraud_probability = float(np.clip(fraud_probability, 0, 1))

pred_is_fraud = fraud_probability >= threshold
tier = risk_tier(fraud_probability)
action = decision_action(tier)

pred_col1, pred_col2 = st.columns([1, 1])
with pred_col1:
    st.markdown("<div class='pred-box'>", unsafe_allow_html=True)
    if pred_is_fraud:
        st.markdown("<p class='risk-high'>FRAUD DETECTED</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='risk-low'>LEGITIMATE</p>", unsafe_allow_html=True)
    action_class = "block" if action in {"Block Transaction", "Flag for Review"} else "approve"
    st.write(f"Fraud Probability: **{fraud_probability * 100:.2f}%**")
    st.write(f"Risk Tier: **{tier}**")
    st.markdown(f"<p class='{action_class}'>Decision Action: {action}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with pred_col2:
    importances = extract_feature_importances(model_obj)
    if importances is not None:
        contrib = np.abs(model_input.values[0]) * importances
        contrib_df = pd.DataFrame({"feature": feature_columns, "contribution": contrib}).sort_values(
            "contribution", ascending=False
        )
    else:
        z = np.abs((model_input.iloc[0] - feature_means) / feature_stds)
        contrib_df = pd.DataFrame({"feature": z.index, "contribution": z.values}).sort_values(
            "contribution", ascending=False
        )
    st.subheader("Top Contributing Features")
    st.bar_chart(contrib_df.head(8).set_index("feature"))

st.header("Section 3 - Detection Method Comparison")
st.image(str(OUTPUTS_DIR / "pr_curve_all_models.png"), caption="Precision-Recall Curves for All 4 Models")
metric_cols = st.columns(3)
metric_cols[0].metric("Selected Model F1", f"{selected_row['F1']:.3f}")
metric_cols[1].metric("Selected Model Recall", f"{selected_row['Recall']:.3f}")
metric_cols[2].metric("Selected Model Precision", f"{selected_row['Precision']:.3f}")
st.dataframe(
    data["model_table"].style.highlight_max(axis=0, color="#14532d").format("{:.3f}"),
    use_container_width=True,
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(data["threshold_df"]["threshold"], data["threshold_df"]["f1"], label="F1", color="#22c55e")
ax.plot(data["threshold_df"]["threshold"], data["threshold_df"]["precision"], label="Precision", color="#3b82f6")
ax.plot(data["threshold_df"]["threshold"], data["threshold_df"]["recall"], label="Recall", color="#ef4444")
ax.set_title("Threshold vs F1 / Recall / Precision")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.legend()
st.pyplot(fig)
plt.close(fig)

if show_smote_impact:
    st.header("Section 4 - SMOTE Impact Visualization")
    sm_col1, sm_col2 = st.columns(2)
    with sm_col1:
        st.image(str(OUTPUTS_DIR / "smote_impact.png"), caption="Class Distribution Before vs After SMOTE")
    with sm_col2:
        st.image(
            str(OUTPUTS_DIR / "f1_vs_smote_sampling_strategy.png"),
            caption="F1 Score vs SMOTE Sampling Strategy",
        )

st.header("Section 5 - Confusion Matrix & Financial Impact")
probs_df = data["model_test_probs"]
y_true = probs_df["y_true"].values
y_score = probs_df[selected_model].values
y_pred = (y_score >= threshold).astype(int)
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax)
ax.set_title(f"Confusion Matrix ({selected_model}, threshold={threshold:.2f})")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticklabels(["Legitimate", "Fraud"])
ax.set_yticklabels(["Legitimate", "Fraud"], rotation=0)
st.pyplot(fig)
plt.close(fig)

tn, fp, fn, tp = cm.ravel()
avg_txn_amount = st.number_input("Average transaction amount for impact estimate (EUR or INR equivalent)", 1.0, 10000.0, 50.0)
savings = tp * avg_txn_amount
missed_loss = fn * avg_txn_amount
friction_cost = fp * avg_txn_amount * 0.02
net_savings = savings - friction_cost

st.write(
    f"At current threshold: catching {tp} fraud transactions, missing {fn}. "
    f"Estimated gross savings: {savings:,.2f}, missed loss: {missed_loss:,.2f}, "
    f"net after friction penalty: {net_savings:,.2f}."
)
