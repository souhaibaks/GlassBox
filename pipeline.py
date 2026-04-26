"""GlassBox pipeline utilities.

This module is intentionally root level so the outer project can run without the package style glassbox/ directory.
"""

import numpy as np


class Pipeline:
    #Chain transformers and, optionally, a final estimator :

    #If the last step has predict(), it is treated as the estimator. 
    #If all steps are transformers, every step participates in fit_transform() and transform()
    

    def __init__(self, steps):
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        if not self.steps:
            raise ValueError("Pipeline steps cannot be empty.")
        for name, obj in self.steps:
            if not hasattr(obj, "fit"):
                raise TypeError(f"Step '{name}' does not have a fit() method.")
        for name, obj in self._transformer_steps:
            if not hasattr(obj, "transform"):
                raise TypeError(f"Step '{name}' does not have a transform() method.")

    @property
    def _has_estimator(self):
        return hasattr(self.steps[-1][1], "predict")

    @property
    def _transformer_steps(self):
        return self.steps[:-1] if self._has_estimator else self.steps

    @property
    def _estimator(self):
        return self.steps[-1][1] if self._has_estimator else None

    def fit(self, X, y=None):
        Xt = np.asarray(X, dtype=float)
        for _, transformer in self._transformer_steps:
            Xt = transformer.fit_transform(Xt, y)
        if self._estimator is not None:
            self._estimator.fit(Xt, y)
        return self

    def transform(self, X):
        Xt = np.asarray(X, dtype=float)
        for _, transformer in self._transformer_steps:
            Xt = transformer.transform(Xt)
        return Xt

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        if self._estimator is None:
            raise RuntimeError("Pipeline has no final estimator.")
        return self._estimator.predict(self.transform(X))

    def score(self, X, y):
        if self._estimator is None:
            raise RuntimeError("Pipeline has no final estimator.")
        return self._estimator.score(self.transform(X), np.asarray(y))

    def get_params(self):
        params = {}
        for name, step in self.steps:
            params[name] = step
            if hasattr(step, "get_params"):
                for key, value in step.get_params().items():
                    params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params):
        step_map = dict(self.steps)
        for key, value in params.items():
            if "__" in key:
                step_name, param_name = key.split("__", 1)
                if step_name not in step_map:
                    raise ValueError(f"Unknown pipeline step: {step_name}")
                step = step_map[step_name]
                if hasattr(step, "set_params"):
                    step.set_params(**{param_name: value})
                else:
                    setattr(step, param_name, value)
            elif key in step_map:
                self.steps = [
                    (name, value if name == key else step)
                    for name, step in self.steps
                ]
            else:
                setattr(self, key, value)
        return self
