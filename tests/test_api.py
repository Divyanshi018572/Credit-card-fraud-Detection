from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_check():
    """Verify the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_valid():
    """Verify the predict endpoint with dummy data."""
    # Dummy features (28 V-features)
    dummy_v = [0.0] * 28
    payload = {
        "amount": 100.0,
        "v_features": dummy_v
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "probability" in data
    assert "risk_tier" in data

def test_predict_endpoint_invalid_schema():
    """Verify that the API rejects invalid data (wrong number of features)."""
    payload = {
        "amount": 100.0,
        "v_features": [0.0] * 5  # Too few features
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
