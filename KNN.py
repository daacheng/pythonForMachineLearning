from sklearn import datasets
from sklearn import neighbors
knn=neighbors.KNeighborsClassifier()#创建分类器
iris=datasets.load_iris()
knn.fit(iris.data,iris.target)#训练分类器
predictLabel=knn.predict([[ 6.2,  3.4,  5.4,  2.2]])#利用分类器做预测
print(predictLabel)