import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.Estimator import Estimator


class Predictor(Estimator):
    def predict(self, X):
        raise NotImplementedError("predict() not implemented.")

    def score(self, X, y):
        raise NotImplementedError("score() not implemented.")

    def fit_predict(self, X_train, y_train, X_test):
        #Convenience: fit on train, predict on test
        self.fit(X_train, y_train)
        return self.predict(X_test)
