import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

class RidgeRegression:
    def __init__(self, lr=0.01, n_iters=1000, alpha=0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) + 2 * self.alpha * self.weights
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

if __name__=="__main__":
    X,y=make_regression(n_samples=300,n_features=5,n_informative=3,noise=20,random_state=42)
    ridge=RidgeRegression()
    ridge.fit(X,y)
    y_pred=ridge.predict(X)
    print("mse",ridge.mse(y,y_pred))
    print("weights",ridge.weights)