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
