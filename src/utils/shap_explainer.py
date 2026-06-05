"""
SHAP-based feature importance explanations for trained models.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


def explain_sklearn_model(
    model,
    X: np.ndarray,
    feature_names: list,
    sample_size: int = 200,
    background_size: int = 50,
) -> pd.DataFrame:
    """
    Compute SHAP values for a sklearn model.
    Returns a DataFrame of mean |SHAP| per feature (feature importance).
    """
    try:
        import shap
    except ImportError:
        raise RuntimeError("Install shap: pip install shap")

    X_sample = X[:sample_size]
    background = shap.kmeans(X[:background_size], k=10)
    explainer  = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)

    importance = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_tree_model(
    model,
    X: np.ndarray,
    feature_names: list,
    sample_size: int = 500,
) -> pd.DataFrame:
    """Fast SHAP for tree-based models (RF, GBM)."""
    try:
        import shap
    except ImportError:
        raise RuntimeError("Install shap: pip install shap")

    # Access the underlying estimator if wrapped in a Pipeline
    estimator = model
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "estimators_") or hasattr(step, "feature_importances_"):
                estimator = step
                # Transform X through pipeline up to this step
                X = model[:-1].transform(X)
                break

    explainer   = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X[:sample_size])

    importance = np.abs(shap_values).mean(axis=0)
    if importance.ndim > 1:
        importance = importance.mean(axis=0)

    return pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def get_shap_summary(
    model,
    X: np.ndarray,
    feature_names: list,
    model_type: str = "tree",
) -> pd.DataFrame:
    """Dispatch to the right SHAP explainer based on model type."""
    if model_type == "tree" or model_type == "random_forest":
        return explain_tree_model(model, X, feature_names)
    elif model_type in ("ridge", "sklearn"):
        return explain_sklearn_model(model, X, feature_names)
    else:
        logger.warning(f"SHAP not implemented for {model_type}. Using sklearn explainer.")
        return explain_sklearn_model(model, X, feature_names)


def top_features(shap_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return shap_df.head(n).reset_index(drop=True)
