import numpy as np
import pandas as pd

# get and split data
data = pd.read_csv('ex1data2.txt', sep=',', header=None)
dataMatrix = data.iloc[:, 0:2]
labelVector = data.iloc[:, 2]
dataLength = len(labelVector)

# normalize data
dataMatrix = (dataMatrix - np.mean(dataMatrix)) / np.std(dataMatrix)

# prepare data and variables
ones = np.ones((dataLength, 1))
dataMatrix = np.hstack((ones, dataMatrix))
learningRate = 0.01
numberOfIterations = 400
theta = np.zeros((3, 1))
labelVector = labelVector[:, np.newaxis]


def computeCostMulti(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2 * dataLength)

#
# J = computeCostMulti(dataMatrix, labelVector, theta)
# print(J)


def gradientDescentMulti(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta -= (alpha / dataLength) * temp
    return theta


theta = gradientDescentMulti(dataMatrix, labelVector, theta, learningRate, numberOfIterations)
