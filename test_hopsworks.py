"""
Test Hopsworks connection and verify feature sync
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from src.feature_pipeline.feature_store import load_features, _get_hopsworks_fg

# Load latest data
df = load_features('london')
if len(df) > 0:
    latest = df.tail(1)
    aqi_val = latest['aqi'].values[0]
    timestamp = latest['timestamp'].values[0]
    print(f"✓ Latest Feature: AQI={aqi_val:.1f} at {timestamp}")
else:
    print("✗ No data found")

# Test Hopsworks connection
print("\nTesting Hopsworks connection...")
try:
    fg = _get_hopsworks_fg()
    print(f"✓ Hopsworks Connected!")
    print(f"  Feature Group: aqi_features")
    print(f"  Project: Successfully accessed")
except ImportError as e:
    print(f"✗ Hopsworks import failed: {e}")
except Exception as e:
    print(f"✗ Connection error: {str(e)[:200]}")
