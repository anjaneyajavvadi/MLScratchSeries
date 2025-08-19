from ElasticNet import ElasticNetRegression
from RidgeRegression import RidgeRegression
from MultiFeatureLinearRegression import MultiFeatureLinearRegressionFromScratch
from LassoRegression import LassoRegression
from sklearn.datasets import make_regression

X,y=make_regression(n_samples=300,n_features=5,n_informative=3,noise=20,random_state=42)

linear=MultiFeatureLinearRegressionFromScratch()
linear.fit(X,y)

print()
print("**********************LINEAR REGRESSION****************************")
print("MSE :",linear.mse(y,linear.predict(X)))
print("Scratch model weights:", linear.weights)
print("Scratch model bias:", linear.bias)


print()
print("**********************LASSO REGRESSION**********************************")
lasso=LassoRegression()
lasso.fit(X,y)
print("MSE ",lasso.mse(y,lasso.predict(X)))
print("Lasso model weights:", lasso.weights)
print("Lasso model bias:", lasso.bias)

print()
print("***********************RIDGE REGRESSION***************************")
ridge=RidgeRegression()
ridge.fit(X,y)
print("MSE :",ridge.mse(y,ridge.predict(X)))
print("Ridge model weights:", ridge.weights)
print("Ridge model bias:", ridge.bias)


print()
print("***********************ELASTIC NET REGRESSION***************************")
elastic=ElasticNetRegression()
elastic.fit(X,y)
print("MSE ",elastic.mse(y,elastic.predict(X)))
print("ElasticNet model weights:", elastic.weights)
print("ElasticNet model bias:", elastic.bias)

