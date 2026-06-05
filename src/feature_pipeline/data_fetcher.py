"""
Fetches raw AQI and weather data from AQICN and OpenWeather APIs.
"""
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    AQICN_API_TOKEN, OPENWEATHER_API_KEY,
    AQICN_BASE_URL, OPENWEATHER_BASE_URL,
    DEFAULT_CITY, CITIES
)

logger = logging.getLogger(__name__)


class AQICNFetcher:
    """Fetches real-time and historical AQI data from AQICN API."""

    def __init__(self, token: str = AQICN_API_TOKEN):
        self.token = token
        self.session = requests.Session()

    def get_current(self, city: str = DEFAULT_CITY) -> dict:
        city_slug = CITIES.get(city, {}).get("aqicn_name", city)
        url = f"{AQICN_BASE_URL}/feed/{city_slug}/?token={self.token}"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data["status"] != "ok":
            raise ValueError(f"AQICN error: {data.get('data', 'unknown')}")
        return self._parse_station(data["data"])

    def get_by_coords(self, lat: float, lon: float) -> dict:
        url = f"{AQICN_BASE_URL}/feed/geo:{lat};{lon}/?token={self.token}"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data["status"] != "ok":
            raise ValueError(f"AQICN error: {data.get('data', 'unknown')}")
        return self._parse_station(data["data"])

    def _parse_station(self, data: dict) -> dict:
        iaqi = data.get("iaqi", {})
        return {
            "timestamp":  datetime.utcnow(),
            "city":       data.get("city", {}).get("name", "unknown"),
            "aqi":        float(data.get("aqi", 0) or 0),
            "pm25":       float(iaqi.get("pm25", {}).get("v", 0) or 0),
            "pm10":       float(iaqi.get("pm10", {}).get("v", 0) or 0),
            "o3":         float(iaqi.get("o3",   {}).get("v", 0) or 0),
            "no2":        float(iaqi.get("no2",  {}).get("v", 0) or 0),
            "so2":        float(iaqi.get("so2",  {}).get("v", 0) or 0),
            "co":         float(iaqi.get("co",   {}).get("v", 0) or 0),
            "temperature": float(iaqi.get("t",   {}).get("v", 0) or 0),
            "humidity":   float(iaqi.get("h",    {}).get("v", 0) or 0),
            "wind_speed": float(iaqi.get("w",    {}).get("v", 0) or 0),
            "pressure":   float(iaqi.get("p",    {}).get("v", 0) or 0),
        }


class OpenWeatherFetcher:
    """Fetches weather and air pollution data from OpenWeather API."""

    def __init__(self, api_key: str = OPENWEATHER_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict) -> dict:
        params["appid"] = self.api_key
        url = f"{OPENWEATHER_BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_current_weather(self, lat: float, lon: float) -> dict:
        data = self._get("weather", {"lat": lat, "lon": lon, "units": "metric"})
        return {
            "timestamp":   datetime.utcfromtimestamp(data["dt"]),
            "temperature": data["main"]["temp"],
            "feels_like":  data["main"]["feels_like"],
            "humidity":    data["main"]["humidity"],
            "pressure":    data["main"]["pressure"],
            "wind_speed":  data["wind"]["speed"],
            "wind_deg":    data["wind"].get("deg", 0),
            "visibility":  data.get("visibility", 10000) / 1000,
            "clouds":      data["clouds"]["all"],
            "weather_main": data["weather"][0]["main"],
        }

    def get_forecast(self, lat: float, lon: float) -> pd.DataFrame:
        """Returns 5-day / 3-hour forecast."""
        data = self._get("forecast", {"lat": lat, "lon": lon, "units": "metric"})
        rows = []
        for item in data["list"]:
            rows.append({
                "timestamp":   datetime.utcfromtimestamp(item["dt"]),
                "temperature": item["main"]["temp"],
                "humidity":    item["main"]["humidity"],
                "pressure":    item["main"]["pressure"],
                "wind_speed":  item["wind"]["speed"],
                "clouds":      item["clouds"]["all"],
                "weather_main": item["weather"][0]["main"],
            })
        return pd.DataFrame(rows)

    def get_air_pollution(self, lat: float, lon: float) -> dict:
        data = self._get("air_pollution", {"lat": lat, "lon": lon})
        components = data["list"][0]["components"]
        aqi_index = data["list"][0]["main"]["aqi"]
        # OpenWeather AQI is 1-5; we store both
        return {
            "timestamp":    datetime.utcnow(),
            "ow_aqi_index": aqi_index,
            "co":           components.get("co", 0),
            "no":           components.get("no", 0),
            "no2":          components.get("no2", 0),
            "o3":           components.get("o3", 0),
            "so2":          components.get("so2", 0),
            "pm25":         components.get("pm2_5", 0),
            "pm10":         components.get("pm10", 0),
            "nh3":          components.get("nh3", 0),
        }

    def get_air_pollution_history(
        self, lat: float, lon: float, start: datetime, end: datetime
    ) -> pd.DataFrame:
        params = {
            "lat":   lat,
            "lon":   lon,
            "start": int(start.timestamp()),
            "end":   int(end.timestamp()),
        }
        data = self._get("air_pollution/history", params)
        rows = []
        for item in data["list"]:
            c = item["components"]
            rows.append({
                "timestamp": datetime.utcfromtimestamp(item["dt"]),
                "ow_aqi":    item["main"]["aqi"],
                "co":        c.get("co", 0),
                "no2":       c.get("no2", 0),
                "o3":        c.get("o3", 0),
                "so2":       c.get("so2", 0),
                "pm25":      c.get("pm2_5", 0),
                "pm10":      c.get("pm10", 0),
            })
        return pd.DataFrame(rows)


class DataFetcher:
    """Unified fetcher — combines AQICN and OpenWeather into one record."""

    def __init__(self):
        self.aqicn = AQICNFetcher()
        self.ow = OpenWeatherFetcher() if OPENWEATHER_API_KEY else None

    def fetch_current(self, city: str = DEFAULT_CITY) -> dict:
        cfg = CITIES.get(city, {"lat": DEFAULT_CITY, "lon": 0})
        lat, lon = cfg.get("lat", 0), cfg.get("lon", 0)

        record = {}
        try:
            record.update(self.aqicn.get_current(city))
        except Exception as e:
            logger.warning(f"AQICN fetch failed: {e}")
            record["aqi"] = None

        if self.ow and lat and lon:
            try:
                weather = self.ow.get_current_weather(lat, lon)
                for k, v in weather.items():
                    if k != "timestamp":
                        record.setdefault(k, v)
            except Exception as e:
                logger.warning(f"OpenWeather fetch failed: {e}")

        record["city"] = city
        record["timestamp"] = record.get("timestamp", datetime.utcnow())
        return record

    def fetch_weather_forecast(self, city: str = DEFAULT_CITY) -> pd.DataFrame:
        if not self.ow:
            return pd.DataFrame()
        cfg = CITIES.get(city, {})
        lat, lon = cfg.get("lat", 0), cfg.get("lon", 0)
        try:
            return self.ow.get_forecast(lat, lon)
        except Exception as e:
            logger.warning(f"Forecast fetch failed: {e}")
            return pd.DataFrame()

    def fetch_history(
        self, city: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Fetch historical air pollution data (OpenWeather only supports this)."""
        if not self.ow:
            logger.warning("No OpenWeather API key — skipping historical fetch.")
            return pd.DataFrame()
        cfg = CITIES.get(city, {})
        lat, lon = cfg.get("lat", 0), cfg.get("lon", 0)
        # OpenWeather history endpoint allows max 1-year lookback
        frames = []
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=7), end)
            try:
                df = self.ow.get_air_pollution_history(lat, lon, current, chunk_end)
                frames.append(df)
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"History chunk {current}–{chunk_end} failed: {e}")
            current = chunk_end
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
