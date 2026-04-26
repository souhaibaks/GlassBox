import numpy as np
from .Transfomer import Transformer


class Imputer(Transformer):
    pass


class SimpleImputer(Imputer):
    #Fill NaNs per column using mean/median/mode

    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.fill_values_ = None  # one fill value per column

    def fit(self, X, y=None):
        #Compute per-column fill values (ignoring NaNs)
        X = np.asarray(X, dtype=float)
        n_cols = X.shape[1] if X.ndim == 2 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.fill_values_ = np.zeros(n_cols)
        for col in range(n_cols):
            col_data = X[:, col]
            valid = col_data[~np.isnan(col_data)]
            if len(valid) == 0:
                self.fill_values_[col] = 0.0
            elif self.strategy == 'mean':
                self.fill_values_[col] = np.mean(valid)
            elif self.strategy == 'median':
                self.fill_values_[col] = np.median(valid)
            elif self.strategy == 'mode':
                values, counts = np.unique(valid, return_counts=True)
                self.fill_values_[col] = values[np.argmax(counts)]
            else:
                raise ValueError(f"Unknown strategy: '{self.strategy}'. "
                                 f"Use 'mean', 'median', or 'mode'.")
        return self

    def transform(self, X):
        #Replace NaN values with the fitted fill values
        if self.fill_values_ is None:
            raise RuntimeError("SimpleImputer has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        was_1d = X.ndim == 1
        if was_1d:
            X = X.reshape(-1, 1)
        X_filled = X.copy()
        for col in range(X_filled.shape[1]):
            mask = np.isnan(X_filled[:, col])
            X_filled[mask, col] = self.fill_values_[col]
        return X_filled.ravel() if was_1d else X_filled

    def get_params(self):
        return {'strategy': self.strategy}


class KNNImputer(Imputer):
    #Impute NaNs using KNN over non-missing features

    def __init__(self, k=5):
        self.k = k
        self.X_train_ = None

    def fit(self, X, y=None):
        self.X_train_ = np.asarray(X, dtype=float).copy()
        return self

    def transform(self, X):
        if self.X_train_ is None:
            raise RuntimeError("KNNImputer has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        X_imputed = X.copy()
        n_cols = X.shape[1]
        for i in range(X.shape[0]):
            for j in range(n_cols):
                if np.isnan(X[i, j]):
                    other_cols = np.arange(n_cols) != j
                    row = X[i, other_cols]
                    distances = []
                    for ii in range(self.X_train_.shape[0]):
                        if not np.isnan(self.X_train_[ii, j]):
                            train_row = self.X_train_[ii, other_cols]
                            # Skip rows with missing context features.
                            if np.any(np.isnan(train_row)) or np.any(np.isnan(row)):
                                continue
                            dist = np.linalg.norm(train_row - row)
                            distances.append((dist, self.X_train_[ii, j]))
                    distances.sort(key=lambda x: x[0])
                    neighbors = [d[1] for d in distances[:self.k]]
                    X_imputed[i, j] = np.mean(neighbors) if neighbors else 0.0
        return X_imputed

    def get_params(self):
        return {'k': self.k}
