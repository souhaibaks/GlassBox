"""K-nearest neighbors models (classifier and regressor)."""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Estimator import Estimator
from models.Predictor import Predictor


class _KNNBase(Predictor, Estimator):
    #Shared fit logic for KNN

    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train_ = None
        self.y_train_ = None
        self.is_fitted = False

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        self.is_fitted = True
        return self

    def _distances(self, X):
        #Return pairwise distances (n_test, n_train)
        X = np.asarray(X, dtype=float)
        p = 1 if self.metric == 'manhattan' else 2
        # Vectorised: shape (n_test, n_train)
        diff = X[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]
        return np.sum(np.abs(diff) ** p, axis=2) ** (1 / p)

    def _neighbor_indices(self, X):
        dists = self._distances(X)
        return np.argsort(dists, axis=1)[:, :self.n_neighbors]

    def get_params(self):
        return {'n_neighbors': self.n_neighbors, 'metric': self.metric}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class KNNClassifier(_KNNBase):
    #KNN classifier using majority vote

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        neighbor_idx = self._neighbor_indices(X)
        result = []
        for row in neighbor_idx:
            labels = self.y_train_[row]
            values, counts = np.unique(labels, return_counts=True)
            result.append(values[np.argmax(counts)])
        return np.array(result)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


class KNNRegressor(_KNNBase):
    #KNN regressor using the mean of neighbor targets

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        neighbor_idx = self._neighbor_indices(X)
        return np.array([np.mean(self.y_train_[row]) for row in neighbor_idx])

    def score(self, X, y):
        #R² score
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


# Legacy alias kept for backward compatibility with existing test files
KNNEstimator = KNNRegressor
