"""Random forest models built from decision trees."""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Estimator import Estimator
from models.Predictor import Predictor
from models.DecisionTree import DecisionTreeClassifier, DecisionTreeRegressor


def _bootstrap_sample(X, y, rng):
    #Draw a bootstrap (replacement) sample from X, y
    n = X.shape[0]
    idx = rng.choice(n, size=n, replace=True)
    return X[idx], y[idx]


def _resolve_max_features(max_features, n_features):
    if max_features == 'sqrt':
        return max(1, int(np.sqrt(n_features)))
    elif max_features == 'log2':
        return max(1, int(np.log2(n_features)))
    elif isinstance(max_features, int):
        return min(max_features, n_features)
    elif isinstance(max_features, float):
        return max(1, int(max_features * n_features))
    else:
        return n_features  # 'all'


# ---------------------------------------------------------------------------
# Random Forest Classifier
# ---------------------------------------------------------------------------

class RandomForestClassifier(Predictor, Estimator):
    #Random forest classifier using bagging and feature subsampling

    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt',
                 criterion='gini', min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees_ = []
        self.feature_subsets_ = []
        self.is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        k = _resolve_max_features(self.max_features, n_features)

        self.trees_ = []
        self.feature_subsets_ = []

        for _ in range(self.n_estimators):
            X_boot, y_boot = _bootstrap_sample(X, y, rng)
            feat_idx = rng.choice(n_features, size=k, replace=False)
            self.feature_subsets_.append(feat_idx)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees_.append(tree)

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        # Collect predictions from every tree
        preds = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees_, self.feature_subsets_)
        ])  # shape: (n_estimators, n_samples)

        # Majority vote
        result = []
        for col in preds.T:
            values, counts = np.unique(col, return_counts=True)
            result.append(values[np.argmax(counts)])
        return np.array(result)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_params(self):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'criterion': self.criterion,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# Random Forest Regressor
# ---------------------------------------------------------------------------

class RandomForestRegressor(Predictor, Estimator):
    #Random forest regressor using bagging and feature subsampling

    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt',
                 min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees_ = []
        self.feature_subsets_ = []
        self.is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        k = _resolve_max_features(self.max_features, n_features)

        self.trees_ = []
        self.feature_subsets_ = []

        for _ in range(self.n_estimators):
            X_boot, y_boot = _bootstrap_sample(X, y, rng)
            feat_idx = rng.choice(n_features, size=k, replace=False)
            self.feature_subsets_.append(feat_idx)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees_.append(tree)

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        preds = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees_, self.feature_subsets_)
        ])
        return np.mean(preds, axis=0)

    def score(self, X, y):
        #R² score
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def get_params(self):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
