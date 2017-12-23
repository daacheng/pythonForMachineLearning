import numpy as np
import pylab as pl
from sklearn import svm
np.random.seed(0)#每次运行程序时保证结果不变，随机值一样
x=np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]#生成特征点列表
y=[0]*20+[1]*20#生成一个label数组，包括20个0,20个1
clf=svm.SVC(kernel='linear')
clf=clf.fit(x,y)
#获取超平面y=ax+b, w0x+w1y+w2=0,y=(-w0/w1)x-w2/w1
w=clf.coef_[0]
a=-w[0]/w[1]
xx=np.linspace(-5,5)
yy=a*xx-clf.intercept_[0]/w[1]

#获取与超平面平行的两条边界
b = clf.support_vectors_[0]
yy_down = a*xx + (b[1] -  a*b[0])
b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1] - a*b[0])

#print "w: ", w
#print "a: ", a

# print "xx: ", xx
# print "yy: ", yy
#print "support_vectors_: ", clf.support_vectors_
#print "clf.coef_: ", clf.coef_

# switching to the generic n-dimensional parameterization of the hyperplan to the 2D-specific equation
# of a line y=a.x +b: the generic w_0x + w_1y +w_3=0 can be rewritten y = -(w_0/w_1) x + (w_3/w_1)


# plot the line, the points, and the nearest vectors to the plane
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
          s=80, facecolors='none')
pl.scatter(x[:, 0], x[:, 1], c=y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()