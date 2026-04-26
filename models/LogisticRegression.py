from __future__ import print_function, division
import numpy as np
import math

class Sigmoid:

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class LogisticRegression:
    #Binary logistic regression classifier

    def __init__(self, learning_rate=0.1, gradient_descent=True):
        #Create a logistic regression model
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        #Initialize weights based on the number of features
        n_features = np.shape(X)[1]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations=4000):
        #Fit the model parameters
        self._initialize_parameters(X)
        for i in range(n_iterations):
            y_pred = self.sigmoid(X.dot(self.param))
            if self.gradient_descent:
                self.param -= self.learning_rate * -(y - y_pred).dot(X)
            else:
                diag_gradient = self.make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        #Predict class labels for X
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred

    def make_diagonal(self, x):
        #Convert a 1-D array into a diagonal matrix
        m = np.zeros((len(x), len(x)))
        for i in range(len(m[0])):
            m[i, i] = x[i]
        return m

    def score(self, X, y):
        #Return accuracy on (X, y)
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
