import numpy as np
from .Transfomer import Transformer


class Scaler(Transformer):
    pass


class MinMaxScaler(Scaler):
    #Scale features to [0, 1]

    def __init__(self):
        self.min_ = None
        self.max_ = None
        self._range = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self._range = self.max_ - self.min_
        # Avoid division-by-zero for constant columns.
        self._range[self._range == 0] = 1.0
        return self

    def transform(self, X):
        if self.min_ is None:
            raise RuntimeError("MinMaxScaler has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        return (X - self.min_) / self._range

    def inverse_transform(self, X):
        return X * self._range + self.min_

    def get_params(self):
        return {}


class StandardScaler(Scaler):
    #Standardize features to mean=0 and std=1

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division-by-zero for constant columns.
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X):
        return X * self.std_ + self.mean_

    def get_params(self):
        return {}