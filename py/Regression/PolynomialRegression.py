import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score

X=6*np.random.rand(300,1)-3
y=0.8*X**2+0.9*X+2+np.random.randn(300,1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#linear regression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Linear Regression results:\n")
print("MSE:",mse)
print("R2 score:",r2)

features=PolynomialFeatures(degree=2,include_bias=False)
X_train_poly=features.fit_transform(X_train)
X_test_poly=features.transform(X_test)

poly_regressor=LinearRegression()
poly_regressor.fit(X_train_poly,y_train)
y_pred=poly_regressor.predict(X_test_poly)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("\nPolynomial Regression results:\n")
print("MSE:",mse)
print("R2 score:",r2)