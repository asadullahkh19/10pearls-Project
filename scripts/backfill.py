"""
Historical backfill script — fetches past data to build a training dataset.

Usage:
    python scripts/backfill.py --city london --days 90

This script:
  1. Fetches historical air pollution data from OpenWeather (up to 1 year back)
  2. Generates engineered features for all historical rows
  3. Stores everything in the local feature store
"""
import logging
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DEFAULT_CITY, CITIES
from src.feature_pipeline.data_fetcher import DataFetcher, OpenWeatherFetcher
from src.feature_pipeline.feature_engineer import engineer_features
from src.feature_pipeline.feature_store import store_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def backfill(city: str = DEFAULT_CITY, days: int = 90):
    logger.info(f"Starting backfill for '{city}' — {days} days.")
    end   = datetime.utcnow()
    start = end - timedelta(days=days)

    fetcher = DataFetcher()
    df = fetcher.fetch_history(city, start=start, end=end)

    if df.empty:
        logger.error(
            "No historical data returned. Check your OPENWEATHER_API_KEY in .env "
            "and ensure the city is configured in src/config.py CITIES dict."
        )
        return

    logger.info(f"Retrieved {len(df)} raw rows from {start.date()} to {end.date()}.")

    # Add city column
    df["city"] = city
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Build synthetic AQI from PM2.5 if we don't have it from AQICN
    if "aqi" not in df.columns and "pm25" in df.columns:
        df["aqi"] = _pm25_to_aqi(df["pm25"])
        logger.info("Derived AQI from PM2.5 values (OpenWeather source).")

    df = df.sort_values("timestamp").reset_index(drop=True)
    engineered = engineer_features(df)
    engineered = engineered.dropna(subset=["aqi"])

    logger.info(f"Engineered {len(engineered)} feature rows. Storing...")
    store_features(engineered, city)
    logger.info("Backfill complete.")


def _pm25_to_aqi(pm25_series: pd.Series) -> pd.Series:
    """
    Convert PM2.5 (μg/m³) to US EPA AQI using breakpoint linear interpolation.
    Reference: https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
    """
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,   51,  100),
        (35.5,  55.4,   101, 150),
        (55.5,  150.4,  151, 200),
        (150.5, 250.4,  201, 300),
        (250.5, 350.4,  301, 400),
        (350.5, 500.4,  401, 500),
    ]

    def _calc(c):
        for cl, ch, il, ih in breakpoints:
            if cl <= c <= ch:
                return round(((ih - il) / (ch - cl)) * (c - cl) + il)
        return 500

    return pm25_series.apply(_calc)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill historical AQI features.")
    parser.add_argument("--city", default=DEFAULT_CITY, choices=list(CITIES.keys()))
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch")
    args = parser.parse_args()
    backfill(city=args.city, days=args.days)
