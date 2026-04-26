import numpy as np


class Transformer:
    #Transformer base API (fit/transform)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self):
        return {}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
