"""Model evaluation utilities — RMSE, MAE, R²."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Any


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100),
        "n_samples": int(len(y_true)),
    }


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "sklearn",
) -> dict:
    if model_type == "lstm":
        y_pred = model.predict(X_test).flatten()
    else:
        y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    return metrics


def print_metrics(name: str, metrics: dict):
    print(f"\n{'='*40}")
    print(f" {name}")
    print(f"{'='*40}")
    print(f"  RMSE : {metrics['rmse']:.2f}")
    print(f"  MAE  : {metrics['mae']:.2f}")
    print(f"  R²   : {metrics['r2']:.4f}")
    print(f"  MAPE : {metrics['mape']:.2f}%")
    print(f"  N    : {metrics['n_samples']}")


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Create a comparison DataFrame from {model_name: metrics} dict."""
    rows = []
    for name, m in results.items():
        rows.append({"model": name, **m})
    df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return df
