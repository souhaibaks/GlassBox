import numpy as np
from .Transfomer import Transformer

class Scaler(Transformer):
    """Base class for scalers."""
    pass

class MinMaxScaler(Scaler):
    """MinMax scaler."""
    def __init__(self):
        """
        Initializes the MinMaxScaler.
        """
        super().__init__()
        self.min_ = None
        self.max_ = None

    def fit(self, X, y=None):
        """
        Fits the scaler on the data.

        Parameters:
            X (array-like): Input data.
            y (array-like, optional): Target data.

        Returns:
            self: Returns the instance itself.
        """
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted scaler.

        Parameters:
            X (array-like): Input data.

        Returns:
            array-like: Transformed data.
        """
        return (X - self.min_) / (self.max_ - self.min_)

class StandardScaler(Scaler):
    """Standard scaler."""
    def __init__(self):
        """
        Initializes the StandardScaler.
        """
        super().__init__()
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        """
        Fits the scaler on the data.

        Parameters:
            X (array-like): Input data.
            y (array-like, optional): Target data.

        Returns:
            self: Returns the instance itself.
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted scaler.

        Parameters:
            X (array-like): Input data.

        Returns:
            array-like: Transformed data.
        """
        return (X - self.mean_) / self.std_