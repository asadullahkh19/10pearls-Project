"""Random Forest regressor wrapper."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(n_estimators: int = 200, max_depth: int = 15) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )),
    ])


def feature_importances(model: Pipeline, feature_names: list) -> dict:
    rf = model.named_steps["rf"]
    importances = rf.feature_importances_
    return dict(sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    ))
