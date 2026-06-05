"""Integration test for inference pipeline (uses demo data)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.alerts import check_forecast_alerts, health_recommendations
from src.training_pipeline.evaluate import compute_metrics


def _fake_forecast(n: int = 72) -> pd.DataFrame:
    from datetime import datetime, timedelta
    base = datetime.utcnow()
    return pd.DataFrame({
        "timestamp":     [base + timedelta(hours=i) for i in range(1, n+1)],
        "predicted_aqi": np.random.uniform(50, 250, n),
        "lower_bound":   np.random.uniform(40, 220, n),
        "upper_bound":   np.random.uniform(60, 280, n),
        "model_used":    "test",
    })


class TestForecastAlerts:
    def test_alerts_returned_for_high_aqi(self):
        fc = _fake_forecast()
        fc["predicted_aqi"] = 200.0   # all rows unhealthy
        alerts = check_forecast_alerts(fc, "london")
        assert len(alerts) > 0

    def test_no_alerts_for_good_aqi(self):
        fc = _fake_forecast()
        fc["predicted_aqi"] = 30.0
        alerts = check_forecast_alerts(fc, "london")
        assert len(alerts) == 0


class TestHealthRecommendations:
    def test_returns_list(self):
        recs = health_recommendations(50)
        assert isinstance(recs, list)
        assert len(recs) >= 1

    def test_hazardous_has_more_recs(self):
        good_recs      = health_recommendations(25)
        hazardous_recs = health_recommendations(400)
        assert len(hazardous_recs) > len(good_recs)


class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([100.0, 80.0, 120.0])
        m = compute_metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["r2"]   == pytest.approx(1.0)

    def test_metrics_keys(self):
        y    = np.array([100.0, 80.0])
        pred = np.array([110.0, 75.0])
        m    = compute_metrics(y, pred)
        for k in ["rmse", "mae", "r2", "mape", "n_samples"]:
            assert k in m

    def test_rmse_positive(self):
        y    = np.array([100.0, 80.0])
        pred = np.array([90.0, 85.0])
        m    = compute_metrics(y, pred)
        assert m["rmse"] > 0
