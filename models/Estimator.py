import numpy as np


class Estimator:
    #Base class for all estimators

    def fit(self, X, y=None):
        raise NotImplementedError("fit() not implemented.")

    def get_params(self):
        raise NotImplementedError("get_params() not implemented.")

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
