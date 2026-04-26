import numpy as np
from typing import Iterator, Tuple

class KFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int = None):
        #K-fold index generator (similar to scikit-learn's `KFold`)
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        #Yield `(train_indices, test_indices)` for each fold
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate((indices[:start], indices[stop:]))
            yield train_indices, test_indices
            current = stop
