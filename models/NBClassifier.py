"""Gaussian Naive Bayes classifier."""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Estimator import Estimator
from models.Predictor import Predictor


class NBClassifier(Predictor, Estimator):
    #Gaussian Naive Bayes with a variance floor for stability

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_priors_ = {}
        self.class_mean_ = {}
        self.class_var_ = {}
        self.is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]

        for cls in self.classes_:
            X_cls = X[y == cls]
            self.class_priors_[cls] = X_cls.shape[0] / n_samples
            self.class_mean_[cls] = np.mean(X_cls, axis=0)
            # Variance + smoothing term (Laplace-style variance floor)
            self.class_var_[cls] = np.var(X_cls, axis=0) + self.var_smoothing

        self.is_fitted = True
        return self

    def _log_likelihood(self, x, cls):
        #Log-likelihood of sample x under Gaussian for class cls
        mean = self.class_mean_[cls]
        var = self.class_var_[cls]
        return np.sum(
            -0.5 * np.log(2 * np.pi * var)
            - 0.5 * ((x - mean) ** 2) / var
        )

    def predict_log_proba(self, X):
        #Return log-posterior for each class
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        log_posteriors = np.zeros((X.shape[0], len(self.classes_)))
        for j, cls in enumerate(self.classes_):
            log_prior = np.log(self.class_priors_[cls])
            log_posteriors[:, j] = np.array([
                log_prior + self._log_likelihood(x, cls) for x in X
            ])
        return log_posteriors

    def predict(self, X):
        log_post = self.predict_log_proba(X)
        indices = np.argmax(log_post, axis=1)
        return self.classes_[indices]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_params(self):
        return {'var_smoothing': self.var_smoothing}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
