"""
AQI alert system — classifies AQI levels and generates warnings.
"""
import logging
from datetime import datetime
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import AQI_LEVELS, ALERT_THRESHOLD

logger = logging.getLogger(__name__)


def classify_aqi(aqi: float) -> str:
    for label, (low, high) in AQI_LEVELS.items():
        if low <= aqi <= high:
            return label
    return "Hazardous"


def aqi_color(aqi: float) -> str:
    if aqi <= 50:   return "#00e400"   # green
    if aqi <= 100:  return "#ffff00"   # yellow
    if aqi <= 150:  return "#ff7e00"   # orange
    if aqi <= 200:  return "#ff0000"   # red
    if aqi <= 300:  return "#8f3f97"   # purple
    return "#7e0023"                   # maroon


def aqi_emoji(aqi: float) -> str:
    if aqi <= 50:   return "🟢"
    if aqi <= 100:  return "🟡"
    if aqi <= 150:  return "🟠"
    if aqi <= 200:  return "🔴"
    if aqi <= 300:  return "🟣"
    return "⚫"


def generate_alert(aqi: float, city: str) -> Optional[dict]:
    level = classify_aqi(aqi)
    if aqi < ALERT_THRESHOLD:
        return None
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "city":      city,
        "aqi":       aqi,
        "level":     level,
        "message":   _alert_message(level, city, aqi),
        "color":     aqi_color(aqi),
    }


def _alert_message(level: str, city: str, aqi: float) -> str:
    messages = {
        "Unhealthy for Sensitive Groups": (
            f"⚠️ Air quality in {city} is unhealthy for sensitive groups "
            f"(AQI {aqi:.0f}). Children, elderly, and people with respiratory "
            "conditions should limit outdoor activity."
        ),
        "Unhealthy": (
            f"🚨 Unhealthy air quality in {city} (AQI {aqi:.0f}). "
            "Everyone may begin to experience health effects. "
            "Limit prolonged outdoor exertion."
        ),
        "Very Unhealthy": (
            f"🚨🚨 Very unhealthy air in {city} (AQI {aqi:.0f}). "
            "Health alert — everyone may experience serious health effects. "
            "Avoid outdoor activity."
        ),
        "Hazardous": (
            f"☠️ HAZARDOUS air quality in {city} (AQI {aqi:.0f}). "
            "Emergency conditions. Stay indoors with windows closed."
        ),
    }
    return messages.get(level, f"Air quality alert in {city}: AQI {aqi:.0f} ({level}).")


def check_forecast_alerts(forecast_df, city: str) -> list:
    """Scan a 72-hour forecast for hazardous periods and return alerts."""
    alerts = []
    for _, row in forecast_df.iterrows():
        alert = generate_alert(row["predicted_aqi"], city)
        if alert:
            alert["timestamp"] = row["timestamp"].isoformat()
            alerts.append(alert)
    return alerts


def health_recommendations(aqi: float) -> list:
    level = classify_aqi(aqi)
    recs = {
        "Good": [
            "Great day to be active outdoors.",
        ],
        "Moderate": [
            "Unusually sensitive people should consider reducing prolonged outdoor exertion.",
        ],
        "Unhealthy for Sensitive Groups": [
            "Sensitive groups should reduce prolonged outdoor exertion.",
            "Keep windows closed and use air purifier indoors.",
            "Wear an N95 mask if going outside.",
        ],
        "Unhealthy": [
            "Limit prolonged outdoor exertion for everyone.",
            "Move heavy outdoor activities indoors.",
            "Wear an N95 mask if going outside.",
            "Use air purifier indoors.",
        ],
        "Very Unhealthy": [
            "Avoid outdoor activity — stay indoors.",
            "Keep all windows and doors closed.",
            "Run air purifier on highest setting.",
            "If you must go out, wear N95/KN95 mask.",
        ],
        "Hazardous": [
            "STAY INDOORS — avoid any outdoor exposure.",
            "Seal gaps in windows and doors.",
            "Run air purifier continuously.",
            "Seek medical attention if experiencing symptoms.",
        ],
    }
    return recs.get(level, ["Monitor air quality conditions."])
