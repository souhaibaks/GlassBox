"""Random search over hyperparameters with cross-validation."""

import numpy as np
from typing import Dict, Any, List, Union


class RandomSearch:
    """Stochastic hyperparameter search with K-fold cross-validation."""

    def __init__(self, model, param_distributions: Dict[str, List[Any]],
                 n_iter: int = 10, cv: int = 5,
                 scoring=None, random_state: Union[int, None] = None):
        self.model = model
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.results_ = []

    def _cv_score(self, X, y, params, rng):
        """Fit model with params using K-Fold CV and return mean score."""
        self.model.set_params(**params)
        n_samples = len(y)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        fold_sizes = np.full(self.cv, n_samples // self.cv, dtype=int)
        fold_sizes[:n_samples % self.cv] += 1
        current = 0
        scores = []
        for fold_size in fold_sizes:
            test_idx = indices[current:current + fold_size]
            train_idx = np.concatenate([indices[:current],
                                        indices[current + fold_size:]])
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            self.model.fit(X_train, y_train)
            if self.scoring:
                s = self.scoring(y_test, self.model.predict(X_test))
            else:
                s = self.model.score(X_test, y_test)
            scores.append(s)
            current += fold_size
        return float(np.mean(scores))

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        self.results_ = []
        self.best_score_ = -np.inf

        for _ in range(self.n_iter):
            params = {
                name: rng.choice(values)
                for name, values in self.param_distributions.items()
            }
            score = self._cv_score(X, y, params, rng)
            self.results_.append({'params': params, 'mean_cv_score': score})
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = dict(params)

        self.model.set_params(**self.best_params_)
        self.model.fit(X, y)
        return self

    def get_best_params(self):
        return self.best_params_

    def get_best_score(self):
        return self.best_score_

    def get_results(self):
        return self.results_
