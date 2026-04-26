
"""Linear and logistic models implemented with NumPy."""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Estimator import Estimator
from models.Predictor import Predictor


# ---------------------------------------------------------------------------
# Linear Regression (OLS, closed-form)
# ---------------------------------------------------------------------------

class LinearRegression(Predictor, Estimator):
    #Ordinary least squares linear regression

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.beta_ = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        # Normal equation: β = (XᵀX)⁻¹ Xᵀy
        self.beta_ = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.beta_

    def score(self, X, y):
        #R² score
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def get_params(self):
        return {'fit_intercept': self.fit_intercept}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    @property
    def coef_(self):
        if self.beta_ is None:
            return None
        return self.beta_[1:] if self.fit_intercept else self.beta_

    @property
    def intercept_(self):
        return float(self.beta_[0]) if self.fit_intercept and self.beta_ is not None else 0.0


# ---------------------------------------------------------------------------
# Gradient Descent Linear Regression
# ---------------------------------------------------------------------------

class GDLinearRegression(Predictor, Estimator):
    #Linear regression trained with gradient descent and learning-rate decay

    def __init__(self, lr=0.01, n_iter=1000, decay=1.0, fit_intercept=True):
        self.lr = lr
        self.n_iter = n_iter
        self.decay = decay
        self.fit_intercept = fit_intercept
        self.weights_ = None
        self.loss_history_ = []
        self.is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.loss_history_ = []

        for t in range(self.n_iter):
            y_pred = X @ self.weights_
            residuals = y_pred - y
            mse = np.mean(residuals ** 2)
            self.loss_history_.append(mse)
            gradient = (2 / n_samples) * X.T @ residuals
            lr_t = self.lr * (self.decay ** t)
            self.weights_ -= lr_t * gradient

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.weights_

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def get_params(self):
        return {'lr': self.lr, 'n_iter': self.n_iter,
                'decay': self.decay, 'fit_intercept': self.fit_intercept}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# Logistic Regression (multi-class softmax, batch SGD)
# ---------------------------------------------------------------------------

class LogisticRegression(Predictor, Estimator):
    #Multiclass logistic regression trained with gradient descent

    def __init__(self, max_iter=1000, lr=0.01, batch_size=64,
                 tol=1e-4, random_state=None):
        self.max_iter = max_iter
        self.lr = lr
        self.batch_size = batch_size
        self.tol = tol
        self.random_state = random_state
        self.weights_ = None
        self.classes_ = None
        self.class_labels_ = {}
        self.loss_history_ = []
        self.is_fitted = False

    # ------------------------------------------------------------------
    @staticmethod
    def _add_bias(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    @staticmethod
    def _softmax(z):
        # Numerically stable softmax
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _one_hot(self, y):
        y_idx = np.array([self.class_labels_[c] for c in y])
        return np.eye(len(self.classes_))[y_idx]

    def _predict_proba_raw(self, X):
        return self._softmax(X @ self.weights_.T)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.class_labels_ = {c: i for i, c in enumerate(self.classes_)}
        X_b = self._add_bias(X)
        y_oh = self._one_hot(y)
        n_samples, n_features = X_b.shape
        n_classes = len(self.classes_)
        self.weights_ = np.zeros((n_classes, n_features))
        self.loss_history_ = []

        for i in range(self.max_iter):
            probs = self._predict_proba_raw(X_b)
            loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-15), axis=1))
            self.loss_history_.append(loss)

            idx = np.random.choice(n_samples, min(self.batch_size, n_samples), replace=False)
            X_batch, y_batch = X_b[idx], y_oh[idx]
            error = y_batch - self._predict_proba_raw(X_batch)
            update = self.lr * error.T @ X_batch
            self.weights_ += update
            if np.abs(update).max() < self.tol:
                break

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        return self._predict_proba_raw(self._add_bias(X))

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return np.array([self.classes_[i] for i in indices])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))

    def get_params(self):
        return {
            'max_iter': self.max_iter, 'lr': self.lr,
            'batch_size': self.batch_size, 'tol': self.tol,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
