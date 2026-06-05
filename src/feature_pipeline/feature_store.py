"""
Feature Store: persists engineered features locally (SQLite + CSV).
Optionally syncs to Hopsworks when credentials are configured.
"""
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FEATURE_STORE_PATH, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT

logger = logging.getLogger(__name__)

DB_PATH = FEATURE_STORE_PATH / "feature_store.db"
FEATURE_STORE_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Local SQLite store
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS aqi_features (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                city      TEXT    NOT NULL,
                timestamp TEXT    NOT NULL,
                data      TEXT    NOT NULL,
                UNIQUE(city, timestamp)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_city_ts ON aqi_features(city, timestamp)"
        )


def save_features(df: pd.DataFrame, city: str):
    """Upsert feature rows into local SQLite store."""
    init_db()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(str)
    with _get_conn() as conn:
        for _, row in df.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO aqi_features (city, timestamp, data)
                VALUES (?, ?, ?)
                """,
                (city, row["timestamp"], row.to_json()),
            )
    logger.info(f"Saved {len(df)} feature rows for {city}.")


def load_features(
    city: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Load features for a city within an optional date range."""
    init_db()
    query = "SELECT data FROM aqi_features WHERE city = ?"
    params: list = [city]
    if start:
        query += " AND timestamp >= ?"
        params.append(str(start))
    if end:
        query += " AND timestamp <= ?"
        params.append(str(end))
    query += " ORDER BY timestamp ASC"

    with _get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return pd.DataFrame()

    # Stored `data` column contains JSON strings; parse safely into series
    import json
    records = [pd.Series(json.loads(r[0])) for r in rows]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def save_features_csv(df: pd.DataFrame, city: str):
    """Backup to CSV (human-readable)."""
    path = FEATURE_STORE_PATH / f"{city}_features.csv"
    if path.exists():
        existing = pd.read_csv(path, parse_dates=["timestamp"])
        df = pd.concat([existing, df]).drop_duplicates(
            subset=["timestamp"], keep="last"
        ).sort_values("timestamp")
    df.to_csv(path, index=False)
    logger.info(f"CSV backup saved: {path}")


# ---------------------------------------------------------------------------
# Hopsworks integration (optional)
# ---------------------------------------------------------------------------

def _get_hopsworks_fg(fg_name: str = "aqi_features", version: int = 1):
    try:
        import hopsworks
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        return fs.get_or_create_feature_group(
            name=fg_name,
            version=version,
            primary_key=["city", "timestamp"],
            description="Hourly AQI and weather features",
            online_enabled=True,
        )
    except ImportError:
        raise RuntimeError("hopsworks package not installed. Run: pip install hopsworks")


def save_to_hopsworks(df: pd.DataFrame):
    if not HOPSWORKS_API_KEY:
        logger.debug("Hopsworks key not set — skipping cloud sync.")
        return
    try:
        fg = _get_hopsworks_fg()
        fg.insert(df, overwrite=False)
        logger.info(f"Synced {len(df)} rows to Hopsworks.")
    except Exception as e:
        logger.warning(f"Hopsworks sync failed: {e}")


def load_from_hopsworks(
    city: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    if not HOPSWORKS_API_KEY:
        return pd.DataFrame()
    try:
        fg = _get_hopsworks_fg()
        query = fg.select_all().filter(fg.city == city)
        df = query.read()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if start:
            df = df[df["timestamp"] >= start]
        if end:
            df = df[df["timestamp"] <= end]
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Hopsworks load failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

def store_features(df: pd.DataFrame, city: str):
    save_features(df, city)
    save_features_csv(df, city)
    save_to_hopsworks(df)


def fetch_features(
    city: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    df = load_from_hopsworks(city, start, end)
    if df.empty:
        df = load_features(city, start, end)
    return df
