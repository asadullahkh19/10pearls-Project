"""
Flask REST API for AQI predictions.
Run: python app/flask_api.py
Endpoints:
    GET /api/health
    GET /api/predict?city=london&model=auto
    GET /api/current?city=london
    GET /api/history?city=london&days=7
    GET /api/alerts?city=london
    GET /api/models?city=london
    GET /api/cities
"""
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.config import CITIES, DEFAULT_CITY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


def _err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


def _city_or_default() -> str:
    city = request.args.get("city", DEFAULT_CITY).lower()
    if city not in CITIES:
        return None
    return city


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.get("/api/cities")
def list_cities():
    return jsonify({"cities": list(CITIES.keys())})


@app.get("/api/current")
def current():
    city = _city_or_default()
    if not city:
        return _err(f"Unknown city. Choose from: {list(CITIES.keys())}")
    try:
        from src.feature_pipeline.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        data    = fetcher.fetch_current(city)
        data["timestamp"] = data["timestamp"].isoformat()
        return jsonify(data)
    except Exception as e:
        logger.exception(e)
        return _err(str(e), 500)


@app.get("/api/predict")
def predict():
    city       = _city_or_default()
    model_name = request.args.get("model", None)
    if model_name == "auto":
        model_name = None
    if not city:
        return _err(f"Unknown city. Choose from: {list(CITIES.keys())}")
    try:
        from src.inference_pipeline.predict import predict_next_72h
        df = predict_next_72h(city=city, model_name=model_name)
        result = df.copy()
        result["timestamp"] = result["timestamp"].astype(str)
        return jsonify({
            "city":      city,
            "model":     result["model_used"].iloc[0] if not result.empty else "unknown",
            "generated": datetime.utcnow().isoformat(),
            "forecast":  result.to_dict(orient="records"),
        })
    except FileNotFoundError as e:
        return _err(str(e), 404)
    except Exception as e:
        logger.exception(e)
        return _err(str(e), 500)


@app.get("/api/history")
def history():
    city = _city_or_default()
    if not city:
        return _err(f"Unknown city. Choose from: {list(CITIES.keys())}")
    days = int(request.args.get("days", 7))
    if days < 1 or days > 365:
        return _err("days must be between 1 and 365.")
    try:
        from src.feature_pipeline.feature_store import load_features
        end   = datetime.utcnow()
        start = end - timedelta(days=days)
        df    = load_features(city, start=start, end=end)
        if df.empty:
            return jsonify({"city": city, "history": [], "n": 0})
        df["timestamp"] = df["timestamp"].astype(str)
        cols = ["timestamp", "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
                "temperature", "humidity", "wind_speed"]
        cols = [c for c in cols if c in df.columns]
        return jsonify({
            "city":    city,
            "n":       len(df),
            "history": df[cols].to_dict(orient="records"),
        })
    except Exception as e:
        logger.exception(e)
        return _err(str(e), 500)


@app.get("/api/alerts")
def alerts():
    city = _city_or_default()
    if not city:
        return _err(f"Unknown city. Choose from: {list(CITIES.keys())}")
    try:
        from src.inference_pipeline.predict import predict_next_72h
        from src.utils.alerts import check_forecast_alerts
        forecast = predict_next_72h(city=city)
        alert_list = check_forecast_alerts(forecast, city)
        return jsonify({"city": city, "alerts": alert_list, "n": len(alert_list)})
    except FileNotFoundError as e:
        return _err(str(e), 404)
    except Exception as e:
        logger.exception(e)
        return _err(str(e), 500)


@app.get("/api/models")
def models():
    city = _city_or_default()
    if not city:
        return _err(f"Unknown city. Choose from: {list(CITIES.keys())}")
    try:
        from src.training_pipeline.model_registry import list_models
        model_list = list_models(city)
        return jsonify({"city": city, "models": model_list})
    except Exception as e:
        logger.exception(e)
        return _err(str(e), 500)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
