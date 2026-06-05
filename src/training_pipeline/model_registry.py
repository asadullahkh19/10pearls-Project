"""
Local model registry: saves/loads trained models with metadata.
Optionally pushes to Hopsworks Model Registry.
"""
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import MODEL_REGISTRY_PATH, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT

logger = logging.getLogger(__name__)
MODEL_REGISTRY_PATH.mkdir(parents=True, exist_ok=True)


def _model_dir(city: str, model_name: str) -> Path:
    d = MODEL_REGISTRY_PATH / city / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model(
    model: Any,
    model_name: str,
    city: str,
    metrics: dict,
    feature_columns: list,
    model_type: str = "sklearn",
) -> Path:
    d = _model_dir(city, model_name)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if model_type == "lstm":
        model_path = d / f"{timestamp}.keras"
        model.save(str(model_path))
    else:
        model_path = d / f"{timestamp}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    meta = {
        "model_name":    model_name,
        "city":          city,
        "model_type":    model_type,
        "timestamp":     timestamp,
        "metrics":       metrics,
        "feature_columns": feature_columns,
        "model_file":    model_path.name,
    }
    meta_path = d / f"{timestamp}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Write "latest" pointer
    latest_path = d / "latest.json"
    with open(latest_path, "w") as f:
        json.dump({"file": str(model_path), "meta": str(meta_path)}, f)

    logger.info(f"Saved {model_name} ({city}) → {model_path}")
    _push_to_hopsworks(model_path, meta)
    return model_path


def load_model(city: str, model_name: str) -> tuple[Any, dict]:
    d = _model_dir(city, model_name)
    latest_path = d / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"No saved model for {model_name}/{city}")

    with open(latest_path) as f:
        latest = json.load(f)

    model_file = Path(latest["file"])
    meta_file  = Path(latest["meta"])

    with open(meta_file) as f:
        meta = json.load(f)

    if meta["model_type"] == "lstm":
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_file))
    else:
        with open(model_file, "rb") as f:
            model = pickle.load(f)

    return model, meta


def list_models(city: Optional[str] = None) -> list[dict]:
    results = []
    search_root = MODEL_REGISTRY_PATH / city if city else MODEL_REGISTRY_PATH
    for meta_file in search_root.rglob("*_meta.json"):
        with open(meta_file) as f:
            results.append(json.load(f))
    return sorted(results, key=lambda x: x["timestamp"], reverse=True)


def _push_to_hopsworks(model_path: Path, meta: dict):
    if not HOPSWORKS_API_KEY:
        return
    try:
        import hopsworks
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        hw_model = mr.sklearn.create_model(
            name=meta["model_name"],
            metrics=meta["metrics"],
            description=f"AQI predictor for {meta['city']}",
        )
        hw_model.save(str(model_path.parent))
        logger.info(f"Pushed {meta['model_name']} to Hopsworks Model Registry.")
    except Exception as e:
        logger.warning(f"Hopsworks model push failed: {e}")
