"""Small EDA helper for numeric arrays."""

import numpy as np
from .Transfomer import Transformer


class EDAInspector(Transformer):
    #Compute per-column statistics, correlations, and IQR-based bounds

    def __init__(self):
        self.stats_ = {}
        self.correlations_ = None
        self.outlier_bounds_ = {}
        self.column_types_ = {}

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self._compute_statistics(X)
        self._compute_correlations(X)
        self._detect_outliers(X)
        self._infer_column_types(X)
        return self

    def transform(self, X):
        #Return a copy with values clipped to IQR bounds
        X = np.asarray(X, dtype=float)
        Xt = X.copy()
        for col, (lo, hi) in self.outlier_bounds_.items():
            Xt[:, col] = np.clip(Xt[:, col], lo, hi)
        return Xt

    def report(self):
        #Return a JSON-serializable summary
        def _to_python(v):
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        stats_out = {
            str(col): {k: _to_python(v) for k, v in s.items()}
            for col, s in self.stats_.items()
        }
        bounds_out = {
            str(col): [_to_python(lo), _to_python(hi)]
            for col, (lo, hi) in self.outlier_bounds_.items()
        }
        corr_out = (self.correlations_.tolist()
                    if self.correlations_ is not None else None)
        types_out = {str(col): t for col, t in self.column_types_.items()}

        return {
            'statistics': stats_out,
            'correlations': corr_out,
            'outlier_bounds': bounds_out,
            'column_types': types_out,
        }

    def _mode(self, data):
        data = data[~np.isnan(data)]
        if len(data) == 0:
            return float('nan')
        values, counts = np.unique(data, return_counts=True)
        return float(values[np.argmax(counts)])

    def _skewness(self, data):
        data = data[~np.isnan(data)]
        if len(data) < 2:
            return 0.0
        m = np.mean(data)
        s = np.std(data)
        return float(np.mean(((data - m) / s) ** 3)) if s > 0 else 0.0

    def _kurtosis(self, data):
        data = data[~np.isnan(data)]
        if len(data) < 2:
            return 0.0
        m = np.mean(data)
        s = np.std(data)
        return float(np.mean(((data - m) / s) ** 4) - 3) if s > 0 else 0.0

    def _compute_statistics(self, X):
        for col in range(X.shape[1]):
            data = X[:, col]
            valid = data[~np.isnan(data)]
            self.stats_[col] = {
                'mean': float(np.mean(valid)) if len(valid) > 0 else float('nan'),
                'median': float(np.median(valid)) if len(valid) > 0 else float('nan'),
                'mode': self._mode(data),
                'std': float(np.std(valid)) if len(valid) > 0 else float('nan'),
                'skewness': self._skewness(data),
                'kurtosis': self._kurtosis(data),
                'n_missing': int(np.sum(np.isnan(data))),
                'n_unique': int(len(np.unique(valid))),
            }

    def _compute_correlations(self, X):
        X_clean = X.copy()
        for col in range(X.shape[1]):
            col_data = X_clean[:, col]
            valid = col_data[~np.isnan(col_data)]
            if len(valid) == 0:
                X_clean[:, col] = 0.0
            else:
                col_mean = np.mean(valid)
                X_clean[np.isnan(col_data), col] = col_mean
        self.correlations_ = np.corrcoef(X_clean.T)

    def _detect_outliers(self, X):
        for col in range(X.shape[1]):
            data = X[:, col]
            valid = data[~np.isnan(data)]
            if len(valid) == 0:
                self.outlier_bounds_[col] = (float('nan'), float('nan'))
                continue
            q1 = np.percentile(valid, 25)
            q3 = np.percentile(valid, 75)
            iqr = q3 - q1
            self.outlier_bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    def _infer_column_types(self, X):
        for col in range(X.shape[1]):
            data = X[:, col]
            valid = data[~np.isnan(data)]
            if len(valid) == 0:
                self.column_types_[col] = 'unknown'
                continue
            unique_vals = np.unique(valid)
            if len(unique_vals) == 2 and set(unique_vals).issubset({0.0, 1.0}):
                self.column_types_[col] = 'boolean'
            elif len(unique_vals) <= 10:
                self.column_types_[col] = 'categorical'
            else:
                self.column_types_[col] = 'numerical'
