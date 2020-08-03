# Modified to fit Python, instead of Octave/MATLAB

# 2
# In this part of this exercise, you will implement linear regression with one variable to predict profits for a food
# truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new
# outlet. The chain already has trucks in various cities and you have data for profits and populations from the
# cities. 3 You would like to use this data to help you select which city to expand to next. The file ex1data1.txt
# contains the dataset for our linear regression problem. The first column is the population of a city and the second
# column is the profit of a food truck in that city. A negative value for profit indicates a loss. The ex1.m script
# has already been set up to load this data for you.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data1.txt', header=None)
dataMatrix = data.iloc[:, 0]
labelVector = data.iloc[:, 1]
dataLength = len(labelVector)

# 2.1 Plotting data
plt.scatter(dataMatrix, labelVector)
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

# 2.2 Gradient Descent
dataMatrix = dataMatrix[:, np.newaxis]
labelVector = labelVector[:, np.newaxis]
theta = np.zeros([2, 1])
iterations = 1500
alpha = 0.01
ones = np.ones((dataLength, 1))
dataMatrix = np.hstack((ones, dataMatrix))


def computeCost(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2 * dataLength)


def gradientDescent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta -= (alpha / dataLength) * temp
    return theta


theta = gradientDescent(dataMatrix, labelVector, theta, alpha, iterations)

J = computeCost(dataMatrix, labelVector, theta)
print(J)

plt.scatter(dataMatrix[:,1], labelVector)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in 10,000s')
plt.plot(dataMatrix[:, 1], np.dot(dataMatrix, theta))
plt.show()

