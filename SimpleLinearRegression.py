import numpy as np
x=[1,3,2,1,3]
y=[14,24,18,17,27]
def fitSLR(x,y):
    n=len(x)
    fenzi=0
    fenmu=0
    for i in range(n):
        fenzi+=(x[i]-np.mean(x))*(y[i]-np.mean(y))#分子
        fenmu+=(x[i]-np.mean(x))**2#分母
    b1=fenzi/fenmu
    b0=np.mean(y)-b1*np.mean(x)
    return b0,b1
def predict(x,b0,b1):
    y=b0+b1*x
    return y
b0,b1=fitSLR(x,y)
print(b0,'###',b1)
y1=predict(6,b0,b1)
print(y1)