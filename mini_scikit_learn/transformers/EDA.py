import numpy as np
from .Transfomer import Transformer

class EDAInspector(Transformer):
    """
    Automated EDA (Exploratory Data Analysis) inspector.

    Provides methods to compute statistics, correlations, detect outliers, and infer column types.
    """

    def __init__(self):
        self.stats_ = {}
        self.correlations_ = None
        self.outlier_bounds_ = {}
        self.column_types_ = {}

    def fit(self, X, y=None):
        """
        Fit the EDA inspector on the data.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).
        y : numpy.ndarray, optional
            Target values, not used.

        Returns:
        --------
        self : object
            Fitted inspector.
        """
        self._compute_statistics(X)
        self._compute_correlations(X)
        self._detect_outliers(X)
        self._infer_column_types(X)
        return self

    def transform(self, X):
        """
        Transform data by capping outliers.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data.

        Returns:
        --------
        X_transformed : numpy.ndarray
            Data with outliers capped.
        """
        X_transformed = X.copy()
        for col in range(X.shape[1]):
            if col in self.outlier_bounds_:
                lower, upper = self.outlier_bounds_[col]
                X_transformed[:, col] = np.clip(X_transformed[:, col], lower, upper)
        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data.
        y : numpy.ndarray, optional
            Target values.

        Returns:
        --------
        X_transformed : numpy.ndarray
            Transformed data.
        """
        return self.fit(X, y).transform(X)

    def report(self):
        """
        Generate a report of the EDA findings.

        Returns:
        --------
        report : dict
            Dictionary containing statistics, correlations, outlier bounds, and column types.
        """
        return {
            'statistics': self.stats_,
            'correlations': self.correlations_,
            'outlier_bounds': self.outlier_bounds_,
            'column_types': self.column_types_
        }

    def _compute_statistics(self, X):
        """
        Compute mean, median, mode, std, skewness, kurtosis for each column.
        """
        for col in range(X.shape[1]):
            data = X[:, col]
            self.stats_[col] = {
                'mean': np.mean(data),
                'median': np.median(data),
                'mode': self._mode(data),
                'std': np.std(data),
                'skewness': self._skewness(data),
                'kurtosis': self._kurtosis(data)
            }

    def _mode(self, data):
        """
        Compute mode of data.
        """
        values, counts = np.unique(data, return_counts=True)
        return values[np.argmax(counts)]

    def _skewness(self, data):
        """
        Compute skewness.
        """
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)

    def _kurtosis(self, data):
        """
        Compute kurtosis.
        """
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3

    def _compute_correlations(self, X):
        """
        Compute Pearson correlation matrix.
        """
        self.correlations_ = np.corrcoef(X.T)

    def _detect_outliers(self, X):
        """
        Detect outliers using IQR method and compute bounds.
        """
        for col in range(X.shape[1]):
            data = X[:, col]
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.outlier_bounds_[col] = (lower, upper)

    def _infer_column_types(self, X):
        """
        Infer column types: numerical, categorical, boolean.
        """
        for col in range(X.shape[1]):
            data = X[:, col]
            unique_vals = np.unique(data)
            bool_values = {0, 1, True, False}
            if len(unique_vals) == 2 and set(unique_vals).issubset(bool_values):
                self.column_types_[col] = 'boolean'
            elif len(unique_vals) < 10:
                self.column_types_[col] = 'categorical'
            else:
                self.column_types_[col] = 'numerical'
