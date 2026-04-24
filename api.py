import json
import time
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from path_utils import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils.config_loader import load_config

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction risk scoring using XGBoost",
    version="1.0.0"
)

# Load config and models at startup
config = load_config()
model = joblib.load(MODELS_DIR / "xgboost_fraud.pkl")
scaler = joblib.load(MODELS_DIR / "log_amount_scaler.pkl")
model_meta = json.loads((MODELS_DIR / "model_metadata.json").read_text())
feature_columns = model_meta["feature_columns"]

# Pydantic Schemas
class TransactionInput(BaseModel):
    amount: float = Field(..., example=150.75, description="Transaction amount in EUR")
    v_features: List[float] = Field(
        ..., 
        min_items=28, 
        max_items=28, 
        description="PCA-transformed features V1 to V28"
    )

class PredictionResponse(BaseModel):
    is_fraud: bool
    probability: float
    risk_tier: str
    action: str
    latency_ms: float

def get_risk_tier(prob: float) -> str:
    tiers = config["business_logic"]["risk_tiers"]
    if prob >= tiers["critical"]: return "Critical"
    if prob >= tiers["high"]: return "High"
    if prob >= tiers["medium"]: return "Medium"
    return "Low"

def get_action(tier: str) -> str:
    if tier in {"Critical", "High"}: return "Block Transaction"
    if tier == "Medium": return "Flag for Review"
    return "Approve"

@app.get("/health")
def health_check():
    """Confirms the API is running and the model is loaded."""
    return {"status": "healthy", "model_version": "xgboost_v1"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: TransactionInput):
    """
    Predicts if a transaction is fraudulent based on input features.
    """
    start_time = time.time()
    
    try:
        # Preprocess input
        log_amount = scaler.transform([[data.amount]])[0][0]
        
        # Combine into DataFrame with correct columns
        input_dict = {f"V{i+1}": val for i, val in enumerate(data.v_features)}
        input_dict["log_Amount"] = log_amount
        input_df = pd.DataFrame([input_dict])[feature_columns]
        
        # Inference
        prob = float(model.predict_proba(input_df)[:, 1][0])
        threshold = config["models"]["xgboost"].get("threshold", model_meta.get("xgb_best_threshold", 0.5))
        
        # Logic
        tier = get_risk_tier(prob)
        
        return {
            "is_fraud": prob >= threshold,
            "probability": round(prob, 4),
            "risk_tier": tier,
            "action": get_action(tier),
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
