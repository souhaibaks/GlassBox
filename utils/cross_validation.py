import numpy as np
import random

def cross_val_score(model, X, y, cv=5, random_state=None):
    #Perform k-fold cross-validation.

    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    X, y = np.asarray(X), np.asarray(y)
    
    if len(X) != len(y):
        raise ValueError("The length of X and y must be equal.")
    
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_sizes = np.full(cv, n_samples // cv, dtype=int)
    fold_sizes[:n_samples % cv] += 1
    current = 0
    scores = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate((indices[:start], indices[stop:]))
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        
        current = stop

    return scores