import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class LinearRegressionFromScratch:
    def __init__(self, lr=0.01, n_iters=10):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = 0 
        self.bias = 0

    def fit(self, X, y):
        X = X.flatten()
        n_samples = X.shape[0]

        for _ in range(self.n_iters):
            y_pred = self.weight * X + self.bias
            dw = (1/n_samples) * np.sum((y_pred - y) * X)
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = X.flatten()
        return self.weight * X + self.bias
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


if __name__=="__main__":
    X,y=make_regression(n_samples=100,n_features=1,noise=20,random_state=42)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    regressor=LinearRegressionFromScratch(lr=0.01,n_iters=1000)
    regressor.fit(X_train,y_train)
    y_pred=regressor.predict(X_test)
    print("model weights and bias of scratch linear regression are:\n")
    print("Weight:", regressor.weight)
    print("Bias:", regressor.bias)

    print("\n")

    skl_regressor=LinearRegression()
    skl_regressor.fit(X_train,y_train)
    y_pred=skl_regressor.predict(X_test)

    print("model weights and bias of sklearn linear regression are:\n")
    print(skl_regressor.coef_)
    print(skl_regressor.intercept_)