"""GlassBox GridSearch with K-Fold cross-validation scoring."""

import numpy as np
import itertools
from typing import Dict, Any, List


class GridSearch:
    """Exhaustive hyperparameter search with cross-validation."""

    def __init__(self, model, param_grid: Dict[str, List[Any]],
                 cv: int = 5, scoring=None):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.results_ = []

    def _cv_score(self, X, y, params):
        """Return mean CV score for one parameter setting."""
        self.model.set_params(**params)
        n_samples = len(y)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
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
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        self.results_ = []
        self.best_score_ = -np.inf

        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            score = self._cv_score(X, y, params)
            self.results_.append({'params': params, 'mean_cv_score': score})
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params

        self.model.set_params(**self.best_params_)
        self.model.fit(X, y)
        return self

    def get_best_params(self):
        return self.best_params_

    def get_best_score(self):
        return self.best_score_

    def get_results(self):
        return self.results_
