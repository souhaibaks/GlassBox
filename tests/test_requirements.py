import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""Requirement-level checks for the GlassBox proposal."""

import sys
sys.path.insert(0, '.')

import numpy as np

from models.DecisionTree import DecisionTreeClassifier
from models.RandomForest import RandomForestClassifier
from models.LinearModel import LogisticRegression
from models.NBClassifier import NBClassifier
from models.KnnEstimator import KNNClassifier
from transformers.Imputers import SimpleImputer
from transformers.Scalers import StandardScaler
from pipeline import Pipeline
from utils import train_test_split


def test_core_models_reach_benchmark_floor():
    """No-sklearn guard for the 90% benchmark target on a simple dataset."""
    np.random.seed(7)
    X0 = np.random.randn(80, 3) + np.array([-2.0, -2.0, 0.0])
    X1 = np.random.randn(80, 3) + np.array([2.0, 2.0, 0.0])
    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    models = [
        DecisionTreeClassifier(max_depth=3),
        RandomForestClassifier(n_estimators=15, max_depth=4, random_state=7),
        LogisticRegression(max_iter=700, lr=0.01, random_state=7),
        NBClassifier(),
        KNNClassifier(n_neighbors=5),
    ]
    scores = []
    for model in models:
        pipe = Pipeline([
            ('imputer', SimpleImputer('mean')),
            ('scaler', StandardScaler()),
            ('model', model),
        ])
        pipe.fit(X_train, y_train)
        scores.append(float(pipe.score(X_test, y_test)))

    assert min(scores) >= 0.9, scores
    print(f"[OK] benchmark floor passed: {[round(s, 3) for s in scores]}")


if __name__ == '__main__':
    test_core_models_reach_benchmark_floor()
    print("\nRequirement checks passed!")
