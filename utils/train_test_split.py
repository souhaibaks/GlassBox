import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Any, Union

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.25, random_state: int = None, stratify: Union[List[Any], np.ndarray, None] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into random train and test subsets (optionally stratified).

    If `stratify` is provided, the split preserves label proportions.
    """
    
    if not isinstance(X, (np.ndarray, list)) or not isinstance(y, (np.ndarray, list)):
        raise ValueError("X and y should be numpy arrays or lists.")
    
    X, y = np.array(X), np.array(y)
    
    if len(X) != len(y):
        raise ValueError("The length of X and y must be equal.")
    
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    if stratify is None:
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        
        n_test = int(len(y) * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    else:
        if len(stratify) != len(y):
            raise ValueError("stratify should be of the same length as y.")
        
        stratified_data = defaultdict(list)
        for idx, label in enumerate(stratify):
            stratified_data[label].append(idx)
        
        train_indices = []
        test_indices = []
        
        for label, indices in stratified_data.items():
            n_test_label = int(len(indices) * test_size)
            random.shuffle(indices)
            
            test_indices.extend(indices[:n_test_label])
            train_indices.extend(indices[n_test_label:])
        
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test
