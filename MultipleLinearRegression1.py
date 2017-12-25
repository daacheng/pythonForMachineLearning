from numpy import genfromtxt
import numpy as np
from sklearn import datasets,linear_model
path=r'D:\daacheng\Python\PythonCode\machineLearning\Delivery.csv'
data=genfromtxt(path,delimiter=',')
print(data)
x=data[:,:-1]
y=data[:,-1]
regr=linear_model.LinearRegression()
regr.fit(x,y)
#y=b0+b1*x1+b2*x2
print(regr.coef_)#b1,b2
print(regr.intercept_)#b0
Xpred=[[102,6]]
Ypred=regr.predict(Xpred)#预测
print(Ypred)