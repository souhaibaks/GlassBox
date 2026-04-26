import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys
sys.path.append('.')
import numpy as np
import transformers.EDA as eda_mod
EDAInspector = eda_mod.EDAInspector

def test_eda_inspector():
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[10, 0] = 10  # outlier
    X[20, 1] = -10  # outlier

    # Fit EDA
    eda = EDAInspector()
    eda.fit(X)

    # Check statistics
    assert 'mean' in eda.stats_[0]
    assert 'median' in eda.stats_[0]
    assert 'mode' in eda.stats_[0]
    assert 'std' in eda.stats_[0]
    assert 'skewness' in eda.stats_[0]
    assert 'kurtosis' in eda.stats_[0]

    # Check correlations
    assert eda.correlations_.shape == (3, 3)

    # Check outlier bounds
    assert 0 in eda.outlier_bounds_
    assert len(eda.outlier_bounds_[0]) == 2

    # Check column types
    assert 0 in eda.column_types_
    assert eda.column_types_[0] == 'numerical'

    # Transform (cap outliers)
    X_transformed = eda.transform(X)
    assert X_transformed.shape == X.shape
    # Check if outliers are capped
    lower, upper = eda.outlier_bounds_[0]
    assert np.all(X_transformed[:, 0] >= lower)
    assert np.all(X_transformed[:, 0] <= upper)

    print("EDA Inspector test passed!")

if __name__ == "__main__":
    test_eda_inspector()