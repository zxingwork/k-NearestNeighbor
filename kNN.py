#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangxing
# datetime:2020/4/19 11:15 上午
# software: PyCharm
from numpy import *
import matplotlib.pyplot as plt
import operator


def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('data/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 9)
        print("the classifier came back with:{}, the real answer is:{}".format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]: errorCount += 1.0
    print("the total error rate is :{}".format(errorCount/float(numTestVecs)))


def autoNorm(dataSet):
    """
    归一化
    :param dataSet: dataSet
    :return: normDataSet, ranges, minVals
    """
    minVals = array(dataSet.min(0))
    maxVals = array(dataSet.max(0))
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    print(normDataSet, ranges, minVals)
    return normDataSet, ranges, minVals


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """

    :param inX: 预测数据集
    :param dataSet: 已知数据集
    :param labels: 已知标签
    :param k: k参数
    :return: 分类
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistanceIndicies = distance.argsort()  # 返回数组从小到大的索引值
    classCount = {}
    for i in range(k):
        voteIlable = labels[sortedDistanceIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    file from data directory to matrix
    :param filename:datafile
    :return matrix, vector
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: 3]
        if listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        index += 1
    return returnMat, array(classLabelVector)


datingClassTest()
