import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""Test: AutoFit end-to-end AutoML."""
import sys
sys.path.insert(0, '.')
import numpy as np
from autofit import AutoFit


def test_autofit_classification():
    """Test AutoFit on synthetic classification data."""
    np.random.seed(42)
    X = np.random.randn(200, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    af = AutoFit(task_type='classification', cv=3, random_state=42)
    af.fit(X, y)
    report = af.report()

    assert report['task_type'] == 'classification'
    assert report['n_samples'] == 200
    assert report['n_features'] == 4
    assert report['best_model'] is not None
    assert report['best_cv_score'] >= 0.5
    assert 'eda' in report
    assert 'evaluation' in report
    print(f"[OK] AutoFit classification: best={report['best_model']}, "
          f"cv_score={report['best_cv_score']:.4f}")
    print(f"  All candidates: {[(r['model'], r['cv_score']) for r in report['all_results']]}")


def test_autofit_regression():
    """Test AutoFit on synthetic regression data."""
    np.random.seed(0)
    X = np.random.randn(150, 3)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 1 + np.random.randn(150) * 0.3

    af = AutoFit(task_type='regression', cv=3, random_state=0)
    af.fit(X, y)
    report = af.report()

    assert report['task_type'] == 'regression'
    assert report['best_model'] is not None
    assert report['evaluation']['r2'] > 0.7
    print(f"[OK] AutoFit regression: best={report['best_model']}, "
          f"R2={report['evaluation']['r2']:.4f}")


def test_autofit_elapsed_time():
    """Verify full AutoML run completes within 120 seconds."""
    import time
    np.random.seed(1)
    X = np.random.randn(200, 5)
    y = (X[:, 0] > 0).astype(int)

    t0 = time.time()
    af = AutoFit(task_type='classification', cv=3, random_state=1)
    af.fit(X, y)
    elapsed = time.time() - t0

    report = af.report()
    assert elapsed < 120, f"AutoFit took {elapsed:.1f}s — exceeds 120s budget!"
    print(f"[OK] AutoFit timing: {elapsed:.1f}s (budget: 120s)")
    print(f"  Report elapsed_seconds={report['elapsed_seconds']}")


if __name__ == '__main__':
    test_autofit_classification()
    test_autofit_regression()
    test_autofit_elapsed_time()
    print("\nAll AutoFit tests passed!")
