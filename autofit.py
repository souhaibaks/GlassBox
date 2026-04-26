"""GlassBox AutoFit - outer project orchestrator.

Runs EDA, preprocessing, hyperparameter search, evaluation, and a compact
JSON-serializable report without depending on the inner ``glassbox/`` package.
"""

import time

import numpy as np

from models import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    KNNClassifier,
    KNNRegressor,
    LinearRegression,
    LogisticRegression,
    NBClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from metrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    classification_report,
)
from model_selection import GridSearch, RandomSearch
from pipeline import Pipeline
from transformers import EDAInspector, SimpleImputer, StandardScaler


def _get_candidate_models(task_type):
    if task_type == "classification":
        return [
            ("DecisionTree", DecisionTreeClassifier(),
             {"max_depth": [3, 5, 10], "criterion": ["gini", "entropy"]}),
            ("RandomForest", RandomForestClassifier(n_estimators=20, random_state=42),
             {"max_depth": [3, 5, 10], "n_estimators": [10, 20]}),
            ("LogisticRegression", LogisticRegression(max_iter=500, random_state=42),
             {"lr": [0.001, 0.01], "batch_size": [32, 64]}),
            ("NaiveBayes", NBClassifier(),
             {"var_smoothing": [1e-9, 1e-7, 1e-5]}),
            ("KNN", KNNClassifier(),
             {"n_neighbors": [3, 5, 7], "metric": ["euclidean", "manhattan"]}),
        ]

    return [
        ("DecisionTree", DecisionTreeRegressor(), {"max_depth": [3, 5, 10]}),
        ("RandomForest", RandomForestRegressor(n_estimators=20, random_state=42),
         {"max_depth": [3, 5, 10], "n_estimators": [10, 20]}),
        ("LinearRegression", LinearRegression(), {"fit_intercept": [True]}),
        ("KNN", KNNRegressor(),
         {"n_neighbors": [3, 5, 7], "metric": ["euclidean", "manhattan"]}),
    ]


class AutoFit:
    #Automated machine-learning orchestrator for numeric matrices  

    def __init__(self, task_type="classification", cv=5, random_state=42,
                 search_strategy="grid", n_iter=10):
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be 'classification' or 'regression'.")
        if search_strategy not in ("grid", "random"):
            raise ValueError("search_strategy must be 'grid' or 'random'.")
        self.task_type = task_type
        self.cv = cv
        self.random_state = random_state
        self.search_strategy = search_strategy
        self.n_iter = n_iter
        self.eda_ = None
        self.pipeline_ = None
        self.best_model_name_ = None
        self.best_model_ = None
        self.best_params_ = None
        self.best_cv_score_ = None
        self.all_results_ = []
        self._report = {}
        self._elapsed = 0.0

    def fit(self, X, y):
        t0 = time.time()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D feature matrix.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.eda_ = EDAInspector()
        self.eda_.fit(X)
        eda_report = self.eda_.report()

        preprocessing = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])
        X_clean = preprocessing.fit_transform(X)
        self.pipeline_ = preprocessing

        best_score = -np.inf
        self.all_results_ = []
        for model_name, model, param_grid in _get_candidate_models(self.task_type):
            search = self._make_search(model, param_grid)
            search.fit(X_clean, y)
            score = search.best_score_
            params = self._json_safe(search.best_params_)
            self.all_results_.append({
                "model": model_name,
                "search_strategy": self.search_strategy,
                "best_params": params,
                "cv_score": round(float(score), 4),
            })
            if score > best_score:
                best_score = score
                self.best_model_name_ = model_name
                self.best_params_ = params
                self.best_model_ = search.model
                self.best_cv_score_ = score

        y_pred = self.best_model_.predict(X_clean)
        if self.task_type == "classification":
            eval_metrics = classification_report(y, y_pred)
        else:
            eval_metrics = {
                "mae": round(MeanAbsoluteError().score(y, y_pred), 4),
                "mse": round(MeanSquaredError().score(y, y_pred), 4),
                "r2": round(R2Score().score(y, y_pred), 4),
            }

        self._elapsed = time.time() - t0
        self._report = {
            "task_type": self.task_type,
            "cv": int(self.cv),
            "search_strategy": self.search_strategy,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "eda": eda_report,
            "best_model": self.best_model_name_,
            "best_params": self.best_params_,
            "best_cv_score": round(float(self.best_cv_score_), 4),
            "all_results": self.all_results_,
            "evaluation": eval_metrics,
            "feature_importance": self._extract_feature_importance(
                self.best_model_, X_clean.shape[1]),
            "elapsed_seconds": round(self._elapsed, 2),
        }
        return self

    def predict(self, X):
        if self.best_model_ is None:
            raise RuntimeError("AutoFit has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        X_clean = self.pipeline_.transform(X) if self.pipeline_ is not None else X
        return self.best_model_.predict(X_clean)

    def report(self):
        return self._report

    def _make_search(self, model, params):
        if self.search_strategy == "random":
            return RandomSearch(model, params, n_iter=self.n_iter, cv=self.cv,
                                random_state=self.random_state)
        return GridSearch(model, params, cv=self.cv)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {k: AutoFit._json_safe(v) for k, v in value.items()}
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    def _extract_feature_importance(self, model, n_features):
        trees = []
        if hasattr(model, "tree_"):
            trees = [(model.tree_, np.arange(n_features))]
        elif hasattr(model, "trees_"):
            subsets = getattr(model, "feature_subsets_",
                              [np.arange(n_features)] * len(model.trees_))
            trees = [(tree.tree_, np.asarray(subset))
                     for tree, subset in zip(model.trees_, subsets)]

        if trees:
            counts = np.zeros(n_features)

            def _count(node, subset):
                if not isinstance(node, tuple):
                    return
                feat, _, left, right = node
                actual = int(subset[feat]) if feat < len(subset) else int(feat)
                if actual < n_features:
                    counts[actual] += 1
                _count(left, subset)
                _count(right, subset)

            for tree, subset in trees:
                _count(tree, subset)
            if counts.sum() > 0:
                return self._rank_importance(counts / counts.sum(),
                                             "split_frequency")

        if hasattr(model, "beta_") and model.beta_ is not None:
            weights = np.asarray(model.beta_).ravel()
            if getattr(model, "fit_intercept", False) and len(weights) == n_features + 1:
                weights = weights[1:]
            if len(weights) == n_features and np.abs(weights).sum() > 0:
                mag = np.abs(weights)
                return self._rank_importance(mag / mag.sum(),
                                             "coefficient_magnitude")

        if hasattr(model, "weights_") and model.weights_ is not None:
            weights = np.asarray(model.weights_)
            if weights.ndim == 2 and weights.shape[1] == n_features + 1:
                weights = weights[:, 1:]
            if weights.ndim == 2 and weights.shape[1] == n_features:
                mag = np.linalg.norm(weights, axis=0)
                if mag.sum() > 0:
                    return self._rank_importance(mag / mag.sum(),
                                                 "weight_magnitude")

        if hasattr(model, "class_mean_") and model.class_mean_:
            means = np.array(list(model.class_mean_.values()))
            if means.ndim == 2 and means.shape[1] == n_features:
                disc = np.var(means, axis=0)
                if disc.sum() > 0:
                    return self._rank_importance(disc / disc.sum(),
                                                 "class_mean_variance")

        return None

    @staticmethod
    def _rank_importance(values, method):
        ranked = sorted(enumerate(values.tolist()), key=lambda item: item[1],
                        reverse=True)
        return [
            {"feature_index": int(index), "importance": round(float(value), 4),
             "method": method}
            for index, value in ranked
        ]
