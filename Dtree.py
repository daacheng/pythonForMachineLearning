from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
import csv
AllElectronics=open('D:\daacheng\Python\PythonCode\machineLearning\AllElectronics.csv','rt')#打开csv文件
readers=csv.reader(AllElectronics)
headers=next(readers)#表头 ['RID', 'age', 'income', 'student', 'credit_rating', 'class_buys_computer']
featureList=[]#特征集合
labelList=[]#标签集合
for row in readers:
    rowDict={}  #每一行数据以集合的形式存储，最后把这些集合存储在列表中
    labelList.append(row[len(row)-1])
    for i in range(1,len(row)-1):
        #print(headers[i])
        #print(row[i])
        rowDict[headers[i]]=row[i]
    featureList.append(rowDict)
vec=DictVectorizer()
dummyX=vec.fit_transform(featureList).toarray()#转换，把特征转换成列表
#names=vec.get_feature_names()
#print(names)
lb=preprocessing.LabelBinarizer()
dummyY=lb.fit_transform(labelList)#对标签进行转换
#dummyY
clf=tree.DecisionTreeClassifier(criterion='entropy')#指定通过信息熵选择节点
clf=clf.fit(dummyX,dummyY)#训练学习
#制造一个新数据来进行预测
oneRowX=dummyX[0,:]
newRowX=[]
newRowX.append(oneRowX)
newRowX[0][0]=1
newRowX[0][1]=0
#print(newRowX[0])
#predictY1=clf.predict(oneRowX)
predictY2=clf.predict(newRowX)#需要传入一个二维数组
#print(predictY1)
print(predictY2)