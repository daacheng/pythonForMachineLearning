import numpy as np
from sklearn import datasets,linear_model
from numpy import genfromtxt
path=r'D:\daacheng\Python\PythonCode\machineLearning\Delivery_Dummy.csv'
data=genfromtxt(path,delimiter=',')
data=data[1:]
x=data[:,:-1]
y=data[:,-1]
print(x)
print(y)
regr=linear_model.LinearRegression()
regr.fit(x,y)
print(regr.coef_)#b1,b2,b3,b4,b5
print(regr.intercept_)#b0