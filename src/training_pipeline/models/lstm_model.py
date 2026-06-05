"""
LSTM model for sequence-based AQI forecasting.
Input shape: (samples, LOOKBACK_HOURS, n_features)
Output: single AQI value (or multi-step with a wrapper)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def build_model(
    n_features: int,
    lookback: int,
    units: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units // 2, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_callbacks(patience: int = 10) -> list:
    return [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]


def create_sequences(
    X: np.ndarray, y: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over the feature matrix to create 3-D sequences."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def create_multistep_sequences(
    X: np.ndarray, y: np.ndarray, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences for multi-step (72-step) direct forecasting."""
    Xs, ys = [], []
    for i in range(lookback, len(X) - horizon + 1):
        Xs.append(X[i - lookback: i])
        ys.append(y[i: i + horizon])
    return np.array(Xs), np.array(ys)


def build_multistep_model(
    n_features: int,
    lookback: int,
    horizon: int = 72,
    units: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(128, activation="relu"),
        layers.Dense(horizon),           # predict all 72 hours at once
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
