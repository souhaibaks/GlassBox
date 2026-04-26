"""GlassBox vs scikit-learn — Comparison

Compares every GlassBox estimator against the equivalent scikit-learn estimator on two standard datasets:
  - Iris  (classification, 150 samples, 4 features, 3 classes)
  - Boston/California Housing  (regression, 506 samples, 13 features)

Metrics reported:
  Classification : accuracy, macro F1
  Regression     : R², RMSE
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── GlassBox imports ──────────────────────────────────────────────────────────
from models.DecisionTree import DecisionTreeClassifier, DecisionTreeRegressor
from models.RandomForest import RandomForestClassifier, RandomForestRegressor
from models.LinearModel import LinearRegression, LogisticRegression
from models.NBClassifier import NBClassifier
from models.KnnEstimator import KNNClassifier, KNNRegressor
from utils import train_test_split, cross_val_score
from metrics.metric import Accuracy, F1Score, MeanSquaredError, R2Score

# ── scikit-learn imports ──────────────────────────────────────────────────────
try:
    from sklearn.tree          import DecisionTreeClassifier  as SK_DTC
    from sklearn.tree          import DecisionTreeRegressor   as SK_DTR
    from sklearn.ensemble      import RandomForestClassifier  as SK_RFC
    from sklearn.ensemble      import RandomForestRegressor   as SK_RFR
    from sklearn.linear_model  import LinearRegression        as SK_LR
    from sklearn.linear_model  import LogisticRegression      as SK_LogR
    from sklearn.naive_bayes   import GaussianNB              as SK_GNB
    from sklearn.neighbors     import KNeighborsClassifier    as SK_KNNC
    from sklearn.neighbors     import KNeighborsRegressor     as SK_KNNR
    from sklearn.datasets      import load_iris, fetch_california_housing
    from sklearn.preprocessing import StandardScaler          as SK_SS
    from sklearn.pipeline      import Pipeline                as SK_Pipe
    from sklearn.model_selection import cross_val_score       as sk_cvs
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARNING] scikit-learn not installed — comparison skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _row(name, gb_acc, sk_acc, metric_name='Accuracy'):
    diff = gb_acc - sk_acc
    return f"  {name:<28} GB={gb_acc:.4f}  SK={sk_acc:.4f}  ({diff:+.4f})"


def _section(title):
    print()
    print('-' * 65)
    print(f'  {title}')
    print('-' * 65)


def gb_cv(model, X, y, cv=5):
    """5-fold CV using GlassBox cross_val_score."""
    scores = cross_val_score(model, X, y, cv=cv, random_state=0)
    return float(np.mean(scores))


def sk_cv(model, X, y, cv=5, scoring='accuracy'):
    """5-fold CV using sklearn cross_val_score."""
    return float(np.mean(sk_cvs(model, X, y, cv=cv, scoring=scoring)))


# ─────────────────────────────────────────────────────────────────────────────
# Classification — Iris
# ─────────────────────────────────────────────────────────────────────────────

def run_classification():
    _section('CLASSIFICATION  Iris Dataset (150 samples, 4 features, 3 classes)')

    iris = load_iris()
    X_raw, y = iris.data, iris.target
    # Standardize for fair comparison (LR / KNN are sensitive to scale)
    from sklearn.preprocessing import StandardScaler as _SS
    X = _SS().fit_transform(X_raw)
    X = X.astype(float)

    print(f"  {'Model':<28} {'GlassBox CV':>14}  {'sklearn CV':>10}  {'Diff':>8}")
    print(f"  {'-'*28} {'-'*14}  {'-'*10}  {'-'*8}")

    pairs = [
        ('DecisionTree (depth=5)',
         DecisionTreeClassifier(max_depth=5, criterion='gini'),
         SK_DTC(max_depth=5, criterion='gini', random_state=42)),
        ('RandomForest (100 trees)',
         RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
         SK_RFC(n_estimators=100, max_depth=5, random_state=42)),
        ('Logistic Regression',
         LogisticRegression(max_iter=2000, lr=0.01, random_state=42),
         SK_LogR(max_iter=2000, random_state=42, solver='lbfgs')),
        ('Naive Bayes',
         NBClassifier(var_smoothing=1e-9),
         SK_GNB(var_smoothing=1e-9)),
        ('KNN (k=5)',
         KNNClassifier(n_neighbors=5, metric='euclidean'),
         SK_KNNC(n_neighbors=5, metric='euclidean')),
    ]

    for name, gb_model, sk_model in pairs:
        gb_score = gb_cv(gb_model, X, y, cv=5)
        sk_score = sk_cv(sk_model, X, y, cv=5, scoring='accuracy')
        print(_row(name, gb_score, sk_score))


# ─────────────────────────────────────────────────────────────────────────────
# Regression — California Housing (subsample for speed)
# ─────────────────────────────────────────────────────────────────────────────

def run_regression():
    _section('REGRESSION  California Housing (500 samples, 8 features)')

    housing = fetch_california_housing()
    # Subsample for speed
    rng = np.random.RandomState(0)
    idx = rng.choice(len(housing.data), size=500, replace=False)
    X_raw = housing.data[idx].astype(float)
    y = housing.target[idx].astype(float)

    from sklearn.preprocessing import StandardScaler as _SS
    X = _SS().fit_transform(X_raw)

    print(f"  {'Model':<28} {'GlassBox R2':>14}  {'sklearn R2':>10}  {'Diff':>8}")
    print(f"  {'-'*28} {'-'*14}  {'-'*10}  {'-'*8}")

    pairs = [
        ('Decision Tree (depth=5)',
         DecisionTreeRegressor(max_depth=5),
         SK_DTR(max_depth=5, random_state=42)),
        ('Random Forest (50 trees)',
         RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
         SK_RFR(n_estimators=50, max_depth=5, random_state=42)),
        ('Linear Regression',
         LinearRegression(),
         SK_LR()),
        ('KNN Regressor (k=5)',
         KNNRegressor(n_neighbors=5),
         SK_KNNR(n_neighbors=5)),
    ]

    for name, gb_model, sk_model in pairs:
        gb_score = gb_cv(gb_model, X, y, cv=5)
        sk_score = sk_cv(sk_model, X, y, cv=5, scoring='r2')
        print(_row(name, gb_score, sk_score, metric_name='R²'))


# ─────────────────────────────────────────────────────────────────────────────
# AutoFit end-to-end demo
# ─────────────────────────────────────────────────────────────────────────────

def run_autofit_demo():
    _section('AUTOFIT  End-to-end AutoML (Iris, 5-fold CV)')
    from autofit import AutoFit
    from sklearn.datasets import load_iris

    iris = load_iris()
    X, y = iris.data.astype(float), iris.target

    af = AutoFit(task_type='classification', cv=5, random_state=42)
    af.fit(X, y)
    rep = af.report()

    print(f"  Best model    : {rep['best_model']}")
    print(f"  Best params   : {rep['best_params']}")
    print(f"  Best CV score : {rep['best_cv_score']:.4f}")
    print(f"  Elapsed       : {rep['elapsed_seconds']}s")
    print()
    print("  All candidates:")
    for r in rep['all_results']:
        print(f"    {r['model']:<22} cv={r['cv_score']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 65)
    print('  GlassBox vs scikit-learn - Benchmark')
    print('=' * 65)

    if not SKLEARN_OK:
        print("scikit-learn not available. Install with: pip install scikit-learn")
        sys.exit(1)

    run_classification()
    run_regression()
    run_autofit_demo()

    print()
    print()
    print('=' * 65)
    print('  Done.')
    print('=' * 65)
