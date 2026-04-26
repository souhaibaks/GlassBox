"""Decision tree models implemented with NumPy."""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Estimator import Estimator
from models.Predictor import Predictor


def _is_leaf(node):
    #Return True if node is a leaf value.
    return not isinstance(node, tuple)


# ---------------------------------------------------------------------------
# Decision Tree Classifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier(Predictor, Estimator):

    def __init__(self, max_depth=None, criterion='gini', min_samples_split=2):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.tree_ = None
        self.n_classes_ = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_classes_ = len(np.unique(y))
        self.tree_ = self._build_tree(X, y, depth=0)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self.tree_) for x in X])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_params(self):
        return {'max_depth': self.max_depth, 'criterion': self.criterion,
                'min_samples_split': self.min_samples_split}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------
    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        if self.criterion == 'gini':
            return 1.0 - np.sum(probs ** 2)
        else:  # entropy
            return -np.sum(probs * np.log2(probs + 1e-15))

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feat, best_thresh = None, None
        parent_imp = self._impurity(y)
        n = len(y)

        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left = y[X[:, feat] < thresh]
                right = y[X[:, feat] >= thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_imp - (
                    len(left) / n * self._impurity(left) +
                    len(right) / n * self._impurity(right)
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh

    def _build_tree(self, X, y, depth):
        max_d = self.max_depth
        # Stopping conditions
        if (max_d is not None and depth >= max_d) or \
                len(np.unique(y)) == 1 or \
                len(y) < self.min_samples_split:
            classes, counts = np.unique(y, return_counts=True)
            return classes[np.argmax(counts)]  # majority class

        feat, thresh = self._best_split(X, y)
        if feat is None:
            classes, counts = np.unique(y, return_counts=True)
            return classes[np.argmax(counts)]

        mask = X[:, feat] < thresh
        left = self._build_tree(X[mask], y[mask], depth + 1)
        right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return (feat, thresh, left, right)

    def _traverse(self, x, node):
        if _is_leaf(node):
            return node
        feat, thresh, left, right = node
        return self._traverse(x, left) if x[feat] < thresh else self._traverse(x, right)


# ---------------------------------------------------------------------------
# Decision Tree Regressor
# ---------------------------------------------------------------------------

class DecisionTreeRegressor(Predictor, Estimator):
    #Decision Tree Regressor using MSE variance reduction


    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.tree_ = self._build_tree(X, y, depth=0)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self.tree_) for x in X])

    def score(self, X, y):
        #R² score
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def get_params(self):
        return {'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------
    def _mse(self, y):
        if len(y) == 0:
            return 0.0
        return float(np.mean((y - np.mean(y)) ** 2))

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feat, best_thresh = None, None
        parent_mse = self._mse(y)
        n = len(y)

        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left = y[X[:, feat] < thresh]
                right = y[X[:, feat] >= thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_mse - (
                    len(left) / n * self._mse(left) +
                    len(right) / n * self._mse(right)
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh

    def _build_tree(self, X, y, depth):
        max_d = self.max_depth
        if (max_d is not None and depth >= max_d) or \
                len(y) < self.min_samples_split or len(y) <= 1:
            return float(np.mean(y))

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return float(np.mean(y))

        mask = X[:, feat] < thresh
        left = self._build_tree(X[mask], y[mask], depth + 1)
        right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return (feat, thresh, left, right)

    def _traverse(self, x, node):
        if _is_leaf(node):
            return node
        feat, thresh, left, right = node
        return self._traverse(x, left) if x[feat] < thresh else self._traverse(x, right)
