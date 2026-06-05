"""
Computes ML features from raw AQI/weather records.
"""
import numpy as np
import pandas as pd
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import LAG_HOURS, ROLLING_WINDOWS, TARGET_COL


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"]        = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek          # 0=Mon, 6=Sun
    df["month"]       = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["is_weekend"]  = (ts.dt.dayofweek >= 5).astype(int)
    df["season"]      = ((ts.dt.month % 12) // 3).astype(int)  # 0=Winter…3=Autumn

    # Cyclical encoding so ML models understand periodicity
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(
    df: pd.DataFrame,
    col: str = TARGET_COL,
    lags: List[int] = LAG_HOURS,
) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    for lag in lags:
        df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    col: str = TARGET_COL,
    windows: List[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    for w in windows:
        df[f"{col}_rolling_mean_{w}h"] = df[col].shift(1).rolling(w).mean()
        df[f"{col}_rolling_std_{w}h"]  = df[col].shift(1).rolling(w).std()
        df[f"{col}_rolling_min_{w}h"]  = df[col].shift(1).rolling(w).min()
        df[f"{col}_rolling_max_{w}h"]  = df[col].shift(1).rolling(w).max()
    return df


def add_change_rate(df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
    df = df.copy()
    df[f"{col}_change_1h"]  = df[col].diff(1)
    df[f"{col}_change_3h"]  = df[col].diff(3)
    df[f"{col}_change_24h"] = df[col].diff(24)
    df[f"{col}_pct_change_1h"] = df[col].pct_change(1).replace([np.inf, -np.inf], 0)
    # Trend direction: +1 rising, -1 falling, 0 stable
    df[f"{col}_trend"] = np.sign(df[f"{col}_change_1h"].fillna(0)).astype(int)
    return df


def add_pollutant_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pollutants = ["pm25", "pm10", "o3", "no2", "so2", "co"]
    available = [p for p in pollutants if p in df.columns]
    if len(available) >= 2:
        df["pollutant_sum"]    = df[available].sum(axis=1)
        df["pollutant_max"]    = df[available].max(axis=1)
        df["pm_ratio"]         = (
            df["pm25"] / (df["pm10"] + 1e-6)
            if "pm25" in df.columns and "pm10" in df.columns
            else 0
        )
    return df


def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "temperature" in df.columns and "humidity" in df.columns:
        # Heat index approximation
        T, H = df["temperature"], df["humidity"]
        df["heat_index"] = (
            -8.78469475556 +
            1.61139411 * T +
            2.33854883889 * H -
            0.14611605 * T * H -
            0.012308094 * T**2 -
            0.0164248277778 * H**2 +
            0.002211732 * T**2 * H +
            0.00072546 * T * H**2 -
            0.000003582 * T**2 * H**2
        )
    if "wind_speed" in df.columns and "pm25" in df.columns:
        # High wind disperses pollutants (inverse relationship)
        df["wind_dispersion"] = df["pm25"] / (df["wind_speed"] + 1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline applied in order."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_change_rate(df)
    df = add_pollutant_interactions(df)
    df = add_weather_interactions(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all feature column names (everything except target and metadata)."""
    exclude = {TARGET_COL, "timestamp", "city"}
    return [c for c in df.columns if c not in exclude]


def create_forecast_targets(
    df: pd.DataFrame, horizons: List[int] = [24, 48, 72]
) -> pd.DataFrame:
    """Add future AQI columns as regression targets for multi-step forecasting."""
    df = df.copy()
    for h in horizons:
        df[f"aqi_t+{h}h"] = df[TARGET_COL].shift(-h)
    return df
