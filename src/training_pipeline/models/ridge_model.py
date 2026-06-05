"""Ridge regression baseline model."""
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def build_model(alpha: float = 1.0, degree: int = 1) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    steps.append(("ridge", Ridge(alpha=alpha, max_iter=10000)))
    return Pipeline(steps)
