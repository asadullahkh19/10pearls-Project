import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# API Keys
AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN", "demo")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# City configuration
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "london")
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", "51.5074"))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", "-0.1278"))

CITIES = {
    "london":   {"lat": 51.5074, "lon": -0.1278, "aqicn_name": "london"},
    "new york": {"lat": 40.7128, "lon": -74.0060, "aqicn_name": "new-york"},
    "beijing":  {"lat": 39.9042, "lon": 116.4074, "aqicn_name": "beijing"},
    "delhi":    {"lat": 28.6139, "lon": 77.2090,  "aqicn_name": "delhi"},
    "paris":    {"lat": 48.8566, "lon": 2.3522,   "aqicn_name": "paris"},
    "karachi":  {"lat": 24.8607, "lon": 67.0011,  "aqicn_name": "karachi"},
    "lahore":   {"lat": 31.5204, "lon": 74.3587,  "aqicn_name": "lahore"},
}

# Local storage paths
FEATURE_STORE_PATH = BASE_DIR / "data" / "processed"
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
MODEL_REGISTRY_PATH = BASE_DIR / "models" / "registry"

# Hopsworks (optional — leave blank to use local storage)
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_predictor")

# Model settings
FORECAST_HOURS = 72          # 3 days ahead
LOOKBACK_HOURS = 48          # Input sequence length for LSTM
TARGET_COL = "aqi"
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48]
ROLLING_WINDOWS = [3, 6, 12, 24]

# AQI thresholds (US EPA standard)
AQI_LEVELS = {
    "Good":        (0,   50),
    "Moderate":    (51,  100),
    "Unhealthy for Sensitive Groups": (101, 150),
    "Unhealthy":   (151, 200),
    "Very Unhealthy": (201, 300),
    "Hazardous":   (301, 500),
}

ALERT_THRESHOLD = 150   # Unhealthy and above triggers alert

# API URLs
AQICN_BASE_URL = "https://api.waqi.info"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
