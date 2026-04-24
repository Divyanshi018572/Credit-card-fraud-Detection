import numpy as np
import pandas as pd
from pipeline.train import compute_metrics

def test_compute_metrics_perfect():
    """Verify metrics for a perfect prediction."""
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2])
    
    metrics = compute_metrics(y_true, y_pred, y_score)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0

def test_compute_metrics_zero():
    """Verify metrics for completely wrong predictions."""
    y_true = pd.Series([1, 1])
    y_pred = np.array([0, 0])
    y_score = np.array([0.1, 0.1])
    
    metrics = compute_metrics(y_true, y_pred, y_score)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
