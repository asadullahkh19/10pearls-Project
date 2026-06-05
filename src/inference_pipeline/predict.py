"""
Inference pipeline: loads the best model and predicts AQI for the next 72 hours.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DEFAULT_CITY, FORECAST_HOURS, LOOKBACK_HOURS, TARGET_COL
from src.feature_pipeline.data_fetcher import DataFetcher
from src.feature_pipeline.feature_engineer import engineer_features, get_feature_columns
from src.feature_pipeline.feature_store import fetch_features
from src.training_pipeline.model_registry import load_model, list_models

logger = logging.getLogger(__name__)


def _best_model_name(city: str) -> str:
    """Pick the model with lowest RMSE from the registry."""
    models = list_models(city)
    if not models:
        raise FileNotFoundError(f"No trained models found for '{city}'. Run training first.")
    return min(models, key=lambda m: m["metrics"].get("rmse", 9999))["model_name"]


def get_recent_features(city: str, hours: int = LOOKBACK_HOURS) -> pd.DataFrame:
    """Load the most recent `hours` of engineered features from the feature store."""
    end   = datetime.utcnow()
    start = end - timedelta(hours=hours + 50)   # buffer for lag computation
    df = fetch_features(city, start=start, end=end)
    if df.empty:
        raise ValueError(f"No recent features for '{city}'. Run feature pipeline first.")
    return df.tail(hours)


def predict_next_72h(
    city: str = DEFAULT_CITY,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        timestamp, predicted_aqi, lower_bound, upper_bound
    covering the next 72 hours.
    """
    if model_name is None:
        model_name = _best_model_name(city)

    model, meta = load_model(city, model_name)
    model_type    = meta["model_type"]
    feature_cols  = meta["feature_columns"]

    df = get_recent_features(city, hours=max(LOOKBACK_HOURS, 72))
    df = engineer_features(df)
    df = df.dropna(subset=feature_cols)

    if len(df) == 0:
        raise ValueError("Not enough engineered features to run inference.")

    rows = []
    base_ts = pd.to_datetime(df["timestamp"].max())

    if model_type == "lstm":
        from src.training_pipeline.models.lstm_model import create_sequences
        X = df[feature_cols].values
        X_seq, _ = create_sequences(X, np.zeros(len(X)), LOOKBACK_HOURS)
        if len(X_seq) == 0:
            raise ValueError("Not enough data for LSTM sequence.")
        last_seq = X_seq[-1:].reshape(1, LOOKBACK_HOURS, len(feature_cols))
        # Autoregressive: feed prediction back as the next input
        current_seq = last_seq.copy()
        for h in range(1, FORECAST_HOURS + 1):
            pred = float(model.predict(current_seq, verbose=0).flatten()[0])
            pred = max(0, pred)
            ts   = base_ts + timedelta(hours=h)
            rows.append({"timestamp": ts, "predicted_aqi": pred})
            # Shift sequence and update last timestep (simplified rollout)
            new_step = current_seq[0, -1:, :].copy()
            # Find aqi column index if present
            if TARGET_COL in feature_cols:
                aqi_idx = feature_cols.index(TARGET_COL)
                new_step[0, aqi_idx] = pred
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, :] = new_step[0]
    else:
        # Sklearn: predict one row at a time using last known features
        last_row = df[feature_cols].iloc[-1].values.reshape(1, -1)
        for h in range(1, FORECAST_HOURS + 1):
            pred = float(model.predict(last_row)[0])
            pred = max(0, pred)
            ts   = base_ts + timedelta(hours=h)
            rows.append({"timestamp": ts, "predicted_aqi": pred})
            # Minimal update: adjust lag_1h column if it exists
            last_row_copy = last_row.copy()
            lag1_key = f"{TARGET_COL}_lag_1h"
            if lag1_key in feature_cols:
                idx = feature_cols.index(lag1_key)
                last_row_copy[0, idx] = pred
            last_row = last_row_copy

    forecast_df = pd.DataFrame(rows)
    # Uncertainty bounds: ±10% (replace with conformal or quantile if needed)
    forecast_df["lower_bound"] = (forecast_df["predicted_aqi"] * 0.90).clip(lower=0)
    forecast_df["upper_bound"] = forecast_df["predicted_aqi"] * 1.10
    forecast_df["model_used"]  = model_name
    return forecast_df


def predict_current(city: str = DEFAULT_CITY) -> dict:
    """Fetch a fresh reading and return current AQI."""
    fetcher = DataFetcher()
    record  = fetcher.fetch_current(city)
    return {
        "city":      city,
        "timestamp": record.get("timestamp", datetime.utcnow()).isoformat(),
        "aqi":       record.get("aqi", 0),
        "pm25":      record.get("pm25", 0),
        "pm10":      record.get("pm10", 0),
        "o3":        record.get("o3", 0),
        "no2":       record.get("no2", 0),
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    df = predict_next_72h(city=args.city, model_name=args.model)
    print(df.to_string(index=False))
