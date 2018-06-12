# k-临近算法（分类）
## 一、算法思路
1. 为了判断未知实例的类别，以所有已知类别的实例作为参考。
2. 选择参数K。
3. 计算未知实例与所有已知实例的距离。
4. 选择距离最近的K个已知实例。
5. 根据少数服从多数，让未知实例归类为K个最邻近样本中最多数的类别。

优点：简单，易于理解，容易实现，通过对K的选择可具备丢噪音数据的强壮性。

缺点：  
1. 需要大量空间存储所有已知实例。
2. 当样本分布不均衡时，比如其中一类样本实例数量过多，占主导的时候，新的未知实例很容易被归类这个主导样本。

改进：考虑距离，根据距离加上权重。

## 二、代码实现

    import numpy as np
    import math
    def createDataset():
        # 构建训练集数据
        dataset = [[0.26547727, 0.27892898,0],
               [0.1337869 , 0.08356665,0],
               [0.02771102, 0.36429227,0],
               [0.81783834, 0.86542639,1],
               [0.99240191, 0.87950623,1],
               [0.99240191, 0.77950623,1]]
        return np.array(dataset)


    def getDistance(instance1,instance2):
        #  计算两点间的距离
        distance=0
        length = len(instance1)
        for i in range(length):
            distance += math.pow(instance1[i]-instance2[i],2)
        return math.sqrt(distance)


    def getNeighbors(trainingSet,testInstance,k):
        # 计算未知实例与所有已知实例的距离。返回最近的K个已知实例
        features = createDataset()[:,:2]
        labels =  createDataset()[:,-1]
        distance_list = []
        for i in range(len(features)):
            distance = getDistance(testInstance,features[i])
            distance_list.append((distance,labels[i]))
        sorted_distance_list = sorted(distance_list)
        neighbors = sorted_distance_list[:k]
        return neighbors


    def countClass(neighbors):
        # 对返回最近的K个已知实例，进行统计分类，根据少数服从多数，让未知实例归类为K个最邻近样本中最多数的类别。
        class_num_dict = {}
        for n in neighbors:
            if n[1] in class_num_dict:
                class_num_dict[n[1]] += 1
            else:
                class_num_dict[n[1]] = 1
        return class_num_dict

    def main():
        trainingSet = createDataset()
        testSet = [[0,0],[1,1],[1.1,1.2]]
        result = []
        for test in testSet:
            # 计算未知实例与所有已知实例的距离。返回最近的K个已知实例
            neighbors = getNeighbors(trainingSet,test,4)
            # 对返回最近的K个已知实例，进行统计分类。
            class_num_dict = countClass(neighbors)
            # 根据少数服从多数，让未知实例归类为K个最邻近样本中最多数的类别。
            result.append(sorted(class_num_dict.items(),key = lambda x:x[1],reverse=True)[0][0])
        print(testSet)
        print(result)

    if __name__ == '__main__':
        main()

## 3、实例训练
场景：有一批相亲网站男生相关信息数据，特征有三个：“每年获得的飞行常客里程数”，“玩视频游戏所耗时间百分比”，“每周消费的冰淇淋公升数”。根据这些特征可以将男生分为“魅力一般的”，“极具魅力的”，“毫无魅力的”三类。现在通过k-临近算法来判断一个男生属于哪一类。

### 3.1、数据样例：

        40920	8.326976	0.953952	3
        14488	7.153469	1.673904	2
        26052	1.441871	0.805124	1
        75136	13.147394	0.428964	1

### 3.2、思路
1. 首先要获取csv文本中的训练集数据。
2. 对训练集数据进行特征缩放，把特征值范围缩小到[-1,1]之间。     (原数据-平均值）/(最大值-最小值)
3. 构建测试数据集。用来验证算法。
4. 对测试集数据进行特征缩放处理。
5. 计算测试集实例与所有训练集实例的距离。返回最近的K个已知实例，及其类别。
6. 对返回最近的K个已知实例，进行统计分类，少数服从多数，确定测试实例属于哪一个类别。

### 3.3、代码实现

    %matplotlib inline
    import csv
    from sklearn import preprocessing
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import math

    def trainingSetExtra():
        # 读取txt文件中的训练集数据，准换成数组，便于进行计算
        datingTestSet = []
        features = []
        labels = []
        with open('datingTestSet2.txt') as f:
            reader = csv.reader(f,delimiter='\t')
            for row in reader:
                newrow = [float(row[0]),float(row[1]),float(row[2]),float(row[3])]
                datingTestSet.append(newrow)
        return np.array(datingTestSet)


    def figureAnalysis(datingTestSet):
        # 用matplotlib画散点图观察
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(datingTestSet[:,0],datingTestSet[:,1],15.0*datingTestSet[:,-1],15.0*datingTestSet[:,-1])
        plt.show()


    def autoNormal(datingTestSet):
        # 特征缩放
        # 因为有的特征数值比较大，比如里程数13400，对结果影响比较大。所以要将特征值的范围转换到[-1,1]之间。
        # 特征缩放公式     （原数据-平均值）/(最大值-最小值)
        # print(datingTestSet[:,:3].min(0))            # 每列最小特征值
        # print(datingTestSet[:,:3].max(0))            # 每列最大特征值
        # print(datingTestSet[:,:3].max(0)-datingTestSet[:,:3].min(0))            # 最大值与最小值之差
        # print(datingTestSet[:,:3].mean(0))           # 每列平均特征值
        datingTestSetFeature = (datingTestSet[:,:3]-datingTestSet[:,:3].mean(0))/(datingTestSet[:,:3].max(0)-datingTestSet[:,:3].min(0))
        datingTestSet[:,:3] = datingTestSetFeature
        print(datingTestSetFeature)
        print(datingTestSet)
        return datingTestSet


    def getDistance(instance1,instance2):
        #  计算两点间的距离
        distance=0
        length = len(instance1)
        for i in range(length):
            distance += math.pow(instance1[i]-instance2[i],2)
        return math.sqrt(distance)


    def getNeighbors(normal_dataset,testInstance,k):
        # 计算未知实例与所有已知实例的距离。返回最近的K个已知实例
        features = normal_dataset[:,:3]
        labels =  normal_dataset[:,-1]
        distance_list = []
        for i in range(len(features)):
            distance = getDistance(testInstance,features[i])
            distance_list.append((distance,labels[i]))
        sorted_distance_list = sorted(distance_list)
        neighbors = sorted_distance_list[:k]
        return neighbors


    def countClass(neighbors):
        # 对返回最近的K个已知实例，进行统计分类，根据少数服从多数，让未知实例归类为K个最邻近样本中最多数的类别。
        class_num_dict = {}
        for n in neighbors:
            if n[1] in class_num_dict:
                class_num_dict[n[1]] += 1
            else:
                class_num_dict[n[1]] = 1
        return class_num_dict


    def main():
        datingTestSet = trainingSetExtra()                                       # 1、获取训练集
        mean = datingTestSet[:,:3].mean(0)                                       # 2、求每一个特征的均值
        max_min_range = datingTestSet[:,:3].max(0)-datingTestSet[:,:3].min(0)    # 3、每个特征最大值与最小值之差
        figureAnalysis(datingTestSet)                                            # 3、可视化分析
        normal_dataset = autoNormal(datingTestSet)                               # 4、特征缩放，把所有特征值范围缩小到[-1,1]之间
                                                                                 # 5、构建测试集数据
        testInstance = [[40920, 8.326976, 0.953952],                    
                       [14488,7.153469,1.673904],
                       [75136,13.147394,0.428964,]]
        res = []
        for i in range(len(testInstance)):
            normal_testInstance = (testInstance[i]-mean)/max_min_range           # 6、对测试集数据进行特征缩放处理
            neighbors = getNeighbors(normal_dataset,normal_testInstance,10)      # 7、计算未知实例与所有已知实例的距离。返回最近的K个已知实例
            class_num_dict = countClass(neighbors)                               # 8、对返回最近的K个已知实例，进行统计分类。
            res_label = sorted(class_num_dict.items(),key = lambda x:x[1],reverse=True)[0][0]
            testInstance[i].append(res_label)
            res.append(testInstance[i])
        print(res)


    if __name__ == '__main__':
        main()

### 3.4、结果
