import csv
import math
import random


# 从数据文件中获取训练集和测试集
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        data = list(lines)
        for x in range(len(data) - 1):
            for y in range(4):
                data[x][y] = float(data[x][y])
            if random.random() < split:
                trainingSet.append(data[x])
            else:
                testSet.append(data[x])


# 计算两个实例间的距离
def getDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += math.pow(instance1[x] - instance2[x], 2)
    return math.sqrt(distance)


# 获取测试集单个实例附近k范围的所有实例
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        d = getDistance(trainingSet[x], testInstance, length)
        distances.append((trainingSet[x], d))
    newDistances = sorted(distances, key=lambda x: x[1])
    neighbors = []
    for x in range(k):
        neighbors.append(newDistances[x][0])
    return neighbors


# 通过获得的K范围内所有实例，获得实例中最多的类别属于哪一类
def getResponse(neighbors):
    classDict = {}  # 定义一个字典用于统计每个类别的个数
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classDict:
            classDict[response] += 1
        else:
            classDict[response] = 1
    newClassDict = sorted(classDict.items(), key=lambda x: x[1], reverse=True)
    return newClassDict


# 计算预测正确的概率
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x][0][0]:
            correct += 1
    return (correct / float(len(testSet)))


def main():
    trainingSet = []
    testSet = []
    split = 0.8
    predictions = []
    loadDataset('D:\daacheng\Python\PythonCode\machineLearning\irisdata.txt', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    k = 5
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
    print(testSet)
    print('-----------------------------------------------')
    print(predictions)
    correct = getAccuracy(testSet, predictions)
    print(correct)


if __name__ == '__main__':
    main()
