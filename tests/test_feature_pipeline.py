"""Unit tests for feature engineering."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_pipeline.feature_engineer import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    add_change_rate,
    engineer_features,
)
from src.utils.alerts import classify_aqi, aqi_color, generate_alert


def _sample_df(n: int = 100) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "city":      "london",
        "aqi":       np.random.uniform(20, 180, n),
        "pm25":      np.random.uniform(5, 80, n),
        "pm10":      np.random.uniform(10, 120, n),
        "o3":        np.random.uniform(10, 80, n),
        "no2":       np.random.uniform(5, 60, n),
        "so2":       np.random.uniform(0, 20, n),
        "co":        np.random.uniform(0, 5, n),
        "temperature": np.random.uniform(-5, 40, n),
        "humidity":    np.random.uniform(30, 95, n),
        "wind_speed":  np.random.uniform(0, 20, n),
        "pressure":    np.random.uniform(990, 1030, n),
    })


class TestTimeFeatures:
    def test_hour_range(self):
        df = add_time_features(_sample_df())
        assert df["hour"].between(0, 23).all()

    def test_day_of_week_range(self):
        df = add_time_features(_sample_df())
        assert df["day_of_week"].between(0, 6).all()

    def test_cyclical_encoding(self):
        df = add_time_features(_sample_df())
        assert df["hour_sin"].between(-1, 1).all()
        assert df["hour_cos"].between(-1, 1).all()

    def test_is_weekend(self):
        df = add_time_features(_sample_df())
        assert df["is_weekend"].isin([0, 1]).all()


class TestLagFeatures:
    def test_lag_columns_created(self):
        df = add_lag_features(_sample_df(), lags=[1, 6, 24])
        assert "aqi_lag_1h"  in df.columns
        assert "aqi_lag_6h"  in df.columns
        assert "aqi_lag_24h" in df.columns

    def test_lag_introduces_nans(self):
        df = add_lag_features(_sample_df(50), lags=[1])
        assert df["aqi_lag_1h"].isna().sum() >= 1


class TestRollingFeatures:
    def test_rolling_columns_created(self):
        df = add_rolling_features(_sample_df(), windows=[3])
        assert "aqi_rolling_mean_3h" in df.columns
        assert "aqi_rolling_std_3h"  in df.columns

    def test_rolling_values_non_negative(self):
        df = add_rolling_features(_sample_df(), windows=[3])
        valid = df["aqi_rolling_mean_3h"].dropna()
        assert (valid >= 0).all()


class TestChangeRate:
    def test_change_columns_created(self):
        df = add_change_rate(_sample_df())
        assert "aqi_change_1h" in df.columns
        assert "aqi_trend"     in df.columns

    def test_trend_values(self):
        df = add_change_rate(_sample_df())
        assert df["aqi_trend"].isin([-1, 0, 1]).all()


class TestEngineerFeatures:
    def test_full_pipeline_runs(self):
        df = engineer_features(_sample_df())
        assert len(df) > 0
        assert "aqi_lag_1h" in df.columns
        assert "hour_sin" in df.columns

    def test_output_length(self):
        raw = _sample_df(50)
        out = engineer_features(raw)
        assert len(out) == len(raw)


class TestAlerts:
    @pytest.mark.parametrize("aqi,expected", [
        (25,  "Good"),
        (75,  "Moderate"),
        (125, "Unhealthy for Sensitive Groups"),
        (175, "Unhealthy"),
        (250, "Very Unhealthy"),
        (400, "Hazardous"),
    ])
    def test_classify_aqi(self, aqi, expected):
        assert classify_aqi(aqi) == expected

    def test_aqi_color_returns_hex(self):
        color = aqi_color(50)
        assert color.startswith("#")

    def test_generate_alert_none_for_good(self):
        assert generate_alert(40, "london") is None

    def test_generate_alert_returns_dict_for_bad(self):
        alert = generate_alert(200, "london")
        assert isinstance(alert, dict)
        assert "message" in alert
        assert "level" in alert
