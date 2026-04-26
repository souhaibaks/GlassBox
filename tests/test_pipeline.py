import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""Test: Pipeline class end-to-end."""
import sys
sys.path.insert(0, '.')
import numpy as np
from transformers.Imputers import SimpleImputer
from transformers.Scalers import StandardScaler
from models.LinearModel import LinearRegression, LogisticRegression
from pipeline import Pipeline


def test_regression_pipeline():
    np.random.seed(0)
    X = np.random.randn(100, 3)
    y = 2 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + np.random.randn(100) * 0.1

    # Introduce some NaN
    X[5, 0] = np.nan
    X[10, 2] = np.nan

    pipe = Pipeline([
        ('imputer', SimpleImputer('mean')),
        ('scaler', StandardScaler()),
        ('model', LinearRegression()),
    ])
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    score = pipe.score(X, y)
    assert score > 0.9, f"Pipeline R2 too low: {score:.3f}"
    assert y_pred.shape == (100,)
    print(f"[OK] Regression pipeline passed: R2={score:.4f}")


def test_classification_pipeline():
    np.random.seed(42)
    X0 = np.random.randn(50, 2)
    X1 = np.random.randn(50, 2) + 3
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=300, lr=0.01, random_state=42)),
    ])
    pipe.fit(X, y)
    acc = pipe.score(X, y)
    assert acc > 0.9, f"Pipeline accuracy too low: {acc:.3f}"
    print(f"[OK] Classification pipeline passed: acc={acc:.4f}")


if __name__ == '__main__':
    test_regression_pipeline()
    test_classification_pipeline()
    print("\nAll pipeline tests passed!")
