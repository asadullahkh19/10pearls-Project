"""
Training pipeline runner — trains all models for a given city.
Run daily via GitHub Actions or manually.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DEFAULT_CITY, CITIES
from src.training_pipeline.train import run_training_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",  default=DEFAULT_CITY, choices=list(CITIES.keys()))
    parser.add_argument("--days",  type=int, default=90)
    parser.add_argument("--no-lstm", action="store_true")
    args = parser.parse_args()
    run_training_pipeline(
        city=args.city,
        lookback_days=args.days,
        train_lstm=not args.no_lstm,
    )
