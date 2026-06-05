"""
Training pipeline: fetches features, trains all models, stores in registry.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    DEFAULT_CITY, FORECAST_HOURS, LOOKBACK_HOURS, TARGET_COL
)
from src.feature_pipeline.feature_store import fetch_features
from src.feature_pipeline.feature_engineer import get_feature_columns
from src.training_pipeline.evaluate import compute_metrics, print_metrics, compare_models
from src.training_pipeline.model_registry import save_model
from src.training_pipeline.models import random_forest_model, ridge_model

# Lazy-import LSTM model (TensorFlow may not be available in all environments)
try:
    from src.training_pipeline.models import lstm_model
except Exception:
    lstm_model = None

logger = logging.getLogger(__name__)


def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """Drop NaN rows and split into X, y arrays."""
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=[TARGET_COL] + feature_cols)
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    return X, y, feature_cols


def train_sklearn_model(
    model,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list,
    city: str,
    n_splits: int = 5,
) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        metrics = compute_metrics(y_val, model.predict(X_val))
        fold_metrics.append(metrics)
        logger.info(f"  fold {fold+1}: RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f}")

    avg = {
        k: float(np.mean([m[k] for m in fold_metrics]))
        for k in ["rmse", "mae", "r2", "mape"]
    }
    avg["n_samples"] = int(y.shape[0])
    print_metrics(f"{model_name} (CV avg)", avg)

    # Final fit on all data
    model.fit(X, y)
    save_model(model, model_name, city, avg, feature_cols, model_type="sklearn")
    return avg


def train_lstm_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list,
    city: str,
    lookback: int = LOOKBACK_HOURS,
    epochs: int = 50,
    batch_size: int = 64,
) -> dict:
    global lstm_model
    # Ensure lstm_model is available (lazy import fallback)
    if lstm_model is None:
        try:
            from src.training_pipeline.models import lstm_model as lm
            lstm_model = lm
        except Exception:
            import subprocess, sys
            logger.warning("TensorFlow/LSTM import failed; attempting to install tensorflow-cpu==2.15.0")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-cpu==2.15.0"])
                from src.training_pipeline.models import lstm_model as lm
                lstm_model = lm
            except Exception as e:
                logger.warning(f"Unable to enable LSTM training: {e}")
                return {}

    X_seq, y_seq = lstm_model.create_sequences(X, y, lookback)
    if len(X_seq) < 100:
        logger.warning("Not enough data for LSTM training (< 100 sequences).")
        return {}

    split = int(len(X_seq) * 0.8)
    X_tr, X_val = X_seq[:split], X_seq[split:]
    y_tr, y_val = y_seq[:split], y_seq[split:]

    model = lstm_model.build_model(
        n_features=X_seq.shape[2], lookback=lookback
    )
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=lstm_model.get_callbacks(patience=10),
        verbose=1,
    )
    metrics = compute_metrics(y_val, model.predict(X_val).flatten())
    print_metrics("LSTM", metrics)
    save_model(model, "lstm", city, metrics, feature_cols, model_type="lstm")
    return metrics


def run_training_pipeline(
    city: str = DEFAULT_CITY,
    lookback_days: int = 90,
    train_lstm: bool = True,
) -> dict:
    logger.info(f"Starting training pipeline for {city}...")
    end   = datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    df = fetch_features(city, start=start, end=end)
    if df.empty or len(df) < 200:
        logger.error(f"Not enough feature data for {city} ({len(df)} rows). Run backfill first.")
        return {}

    logger.info(f"Loaded {len(df)} feature rows.")
    X, y, feature_cols = prepare_data(df)

    results = {}

    # Ridge Regression
    logger.info("Training Ridge Regression...")
    ridge = ridge_model.build_model(alpha=10.0)
    results["ridge"] = train_sklearn_model(ridge, "ridge", X, y, feature_cols, city)

    # Random Forest
    logger.info("Training Random Forest...")
    rf = random_forest_model.build_model()
    results["random_forest"] = train_sklearn_model(rf, "random_forest", X, y, feature_cols, city)

    # LSTM (optional — needs more data)
    if train_lstm and len(X) > LOOKBACK_HOURS * 3:
        logger.info("Training LSTM...")
        results["lstm"] = train_lstm_model(X, y, feature_cols, city)

    comparison = compare_models(results)
    logger.info(f"\nModel comparison:\n{comparison.to_string()}")
    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--no-lstm", action="store_true")
    args = parser.parse_args()
    run_training_pipeline(
        city=args.city,
        lookback_days=args.days,
        train_lstm=not args.no_lstm,
    )
