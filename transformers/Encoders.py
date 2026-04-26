import numpy as np
from .Transfomer import Transformer


class Encoder(Transformer):
    pass


class OrdinalEncoder(Encoder):
    #Encode categories as integers (unknown -> -1)

    def __init__(self, ignore_unknown=True):
        self.ignore_unknown = ignore_unknown
        self.mapping_ = {}

    def fit(self, X, y=None):
        #Learn category -> integer mapping
        X = np.asarray(X).ravel()
        unique_vals = []
        seen = set()
        for v in X:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            key = str(v)
            if key not in seen:
                seen.add(key)
                unique_vals.append(v)
        self.mapping_ = {str(v): idx for idx, v in enumerate(unique_vals)}
        return self

    def transform(self, X):
        X = np.asarray(X).ravel()
        return np.array([self.mapping_.get(str(v), -1) for v in X], dtype=float)

    def get_params(self):
        return {'ignore_unknown': self.ignore_unknown}


class LabelEncoder(Encoder):
    #Encode labels to integers 0..N-1 (keeps `classes_`)

    def __init__(self):
        self.classes_ = None
        self.class_to_index_ = {}

    def fit(self, X, y=None):
        #Learn classes and label -> index mapping
        X = np.asarray(X).ravel()
        unique_vals = []
        seen = set()
        for v in X:
            key = str(v)
            if key not in seen:
                seen.add(key)
                unique_vals.append(v)
        self.classes_ = np.array(unique_vals)
        self.class_to_index_ = {str(v): i for i, v in enumerate(unique_vals)}
        return self

    def transform(self, X):
        X = np.asarray(X).ravel()
        return np.array([self.class_to_index_[str(v)] for v in X], dtype=int)

    def inverse_transform(self, X):
        index_to_class = {i: v for i, v in enumerate(self.classes_)}
        return np.array([index_to_class[int(v)] for v in X])

    def get_params(self):
        return {}


class OneHotEncoder(Encoder):
    #One-hot encode a 1-D categorical array (dense output)

    def __init__(self, sparse=False):
        self.sparse = sparse
        self.categories_ = None
        self._cat_to_idx = {}

    def fit(self, X, y=None):
        X = np.asarray(X).ravel()
        unique_vals = []
        seen = set()
        for v in X:
            key = str(v)
            if key not in seen:
                seen.add(key)
                unique_vals.append(v)
        self.categories_ = np.array(unique_vals)
        self._cat_to_idx = {str(v): i for i, v in enumerate(unique_vals)}
        return self

    def transform(self, X):
        X = np.asarray(X).ravel()
        result = np.zeros((len(X), len(self.categories_)), dtype=float)
        for i, item in enumerate(X):
            idx = self._cat_to_idx.get(str(item), None)
            if idx is not None:
                result[i, idx] = 1.0
        return result

    @property
    def n_categories_(self):
        return len(self.categories_) if self.categories_ is not None else 0

    def get_params(self):
        return {'sparse': self.sparse}
