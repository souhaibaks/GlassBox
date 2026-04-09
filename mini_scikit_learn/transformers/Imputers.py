import numpy as np
import pandas as pd
from .Transfomer import Transformer
import LinearModel


class Imputer(Transformer):
    """Base class for imputers."""
    pass

class SimpleImputer(Imputer):
    """Simple imputer."""
    def __init__(self):
        """
        Initializes the SimpleImputer.
        """
        super().__init__()
        self.fill_value = None

    def fit(self, X, y=None, strategy='median'):
        """
        Fits the imputer on the data using the specified strategy.

        Parameters:
            X (array-like): Input data.
            y (array-like, optional): Target data.
            strategy (str): Strategy for imputation ('mean', 'median', 'mode').

        Returns:
            self: Returns the instance itself.
        """
        if strategy == 'mean':
            self.fill_value = np.nanmean(X)
        elif strategy == 'median':
            self.fill_value = np.nanmedian(X)
        elif strategy == 'mode':
            self.fill_value = pd.Series(X).mode().iloc[0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted imputer.

        Parameters:
            X (array-like): Input data.

        Returns:
            array-like: Transformed data.
        """
        return np.where(pd.isnull(X), self.fill_value, X)
    
class IterativeImputer(Imputer):
    """Iterative imputer."""
    def __init__(self):
        """
        Initializes the IterativeImputer.
        """
        super().__init__()
        self.estimator = None

    def fit(self, X, y=None, estimator=None):
        """
        Fits the imputer on the data using the specified estimator.

        Parameters:
            X (array-like): Input data.
            y (array-like, optional): Target data.
            estimator (object): Estimator to use for imputation.

        Returns:
            self: Returns the instance itself.
        """
        if estimator is None:
            estimator = LinearModel.LinearRegression()
        self.estimator = estimator
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted estimator.

        Parameters:
            X (array-like): Input data.

        Returns:
            array-like: Transformed data.
        """
        return self.estimator.predict(X)

class KNNImputer(Imputer):
    """KNN imputer."""
    def __init__(self):
        """
        Initializes the KNNImputer.
        """
        super().__init__()
        self.k = None
        self.X_train = None

    def fit(self, X, y=None, k=5):
        """
        Fits the imputer on the data using k-nearest neighbors.

        Parameters:
            X (array-like): Input data.
            y (array-like, optional): Target data.
            k (int): Number of neighbors to use for imputation.

        Returns:
            self: Returns the instance itself.
        """
        self.k = k
        self.X_train = X.copy()
        return self
    
    def transform(self, X):
        """
        Transforms the data using k-nearest neighbors.

        Parameters:
            X (array-like): Input data.

        Returns:
            array-like: Transformed data.
        """
        X_imputed = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if np.isnan(X[i, j]):
                    # Find k nearest neighbors based on other features
                    distances = []
                    for ii in range(self.X_train.shape[0]):
                        if not np.isnan(self.X_train[ii, j]):
                            dist = np.linalg.norm(self.X_train[ii, np.arange(X.shape[1]) != j] - X[i, np.arange(X.shape[1]) != j])
                            distances.append((dist, self.X_train[ii, j]))
                    distances.sort()
                    neighbors = [d[1] for d in distances[:self.k]]
                    X_imputed[i, j] = np.mean(neighbors)
        return X_imputed
