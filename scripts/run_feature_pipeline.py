"""
Hourly feature pipeline — fetches current data and stores engineered features.
Run manually or via GitHub Actions / Airflow every hour.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DEFAULT_CITY, CITIES
from src.feature_pipeline.data_fetcher import DataFetcher
from src.feature_pipeline.feature_engineer import engineer_features
from src.feature_pipeline.feature_store import store_features, load_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(city: str = DEFAULT_CITY):
    logger.info(f"Feature pipeline started for: {city}")
    fetcher = DataFetcher()

    # Fetch current reading
    record = fetcher.fetch_current(city)
    if not record.get("aqi"):
        logger.warning("AQI data unavailable — skipping this run.")
        return

    logger.info(f"Fetched AQI={record['aqi']} at {record['timestamp']}")

    # Load recent history to compute lag/rolling features properly
    recent = load_features(city)
    import pandas as pd
    new_row = pd.DataFrame([record])

    if not recent.empty:
        combined = pd.concat([recent, new_row], ignore_index=True)
    else:
        combined = new_row

    combined = combined.sort_values("timestamp").drop_duplicates("timestamp")
    engineered = engineer_features(combined)

    # Store only the latest row
    latest = engineered.tail(1)
    store_features(latest, city)
    logger.info(f"Stored {len(latest)} engineered row(s) for {city}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--city", default=DEFAULT_CITY,
        choices=list(CITIES.keys()) + ["all"],
        help="City to run pipeline for, or 'all'",
    )
    args = parser.parse_args()

    if args.city == "all":
        for city in CITIES:
            try:
                run_pipeline(city)
            except Exception as e:
                logger.error(f"Pipeline failed for {city}: {e}")
    else:
        run_pipeline(args.city)
