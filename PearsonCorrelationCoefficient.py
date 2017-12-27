import numpy as np
import math
#皮尔逊相关系数
def getPearson(x,y):
    xBar=np.mean(x)
    yBar=np.mean(y)
    fenzi=0
    fenmu=0
    x_2=0
    y_2=0
    for i in range(len(x)):
        x_=x[i]-xBar
        y_=y[i]-yBar
        fenzi+=x_*y_
        x_2+=x_**2
        y_2+=y_**2
    fenmu=math.sqrt(x_2*y_2)
    return fenzi/fenmu

#多元线性回归的R平方值（相关系数） degree参数只X的次方
def polyfit(x,y,degree):
    result={}
    coeffs=np.polyfit(x,y,degree)#np的方法直接求得线性相关的系数[b0,b1……bn]
    result['polynomial']=coeffs.tolist()
    p=np.poly1d(coeffs)#p=2.657 x + 5.322  预估的直线
    yhat=p(x)#y的预估值
    y_=np.mean(y)
    ssr=np.sum((yhat-y_)**2)
    sst=np.sum((y-y_)**2)
    result['determination']=ssr/sst
    return result


x=[1,3,8,7,9]
y=[10,12,24,21,34]

#p=getPearson(x,y)
#print(p)#皮尔逊相关系数：衡量两个值线性相关强度的量
#print(p**2)#简单线性回归的R平方值 决定系数 反应因变量的全部变异能通过回归关系被自变量解释的比例

polyfit(x,y,1)