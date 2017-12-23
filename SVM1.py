from sklearn import svm
x=[[2,0],[1,1],[2,3]]#特征向量
y=[0,0,1]#特征对应的label
clf=svm.SVC(kernel='linear')
clf.fit(x,y)
print(clf)
print(clf.support_vectors_)#支持向量
print(clf.support_)#支持向量在x中的索引
print(clf.n_support_)#针对每个标签label找到了几个支持向量
print(clf.predict([[2,0]]))#预测这个点属于哪一类标签label