import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = pd.read_csv('ex2data1.txt', header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, 2]

mask = y == 1
adm = plt.scatter(X[mask][0].values, X[mask][1].values)
not_adm = plt.scatter(X[~mask][0].values, X[~mask][1].values)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))

plt.show()
(m, n) = X.shape


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost(theta, X, y):
    J = (-1 / m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta)))
                          + np.multiply((1 - y), np.log(1 - sigmoid(X @ theta))))
    return J


def gradient(theta, X, y):
    return (1 / m) * X.T @ (sigmoid(X @ theta) - y)


X = np.hstack((np.ones((m, 1)), X))
y = y[:, np.newaxis]
theta = np.zeros((n + 1, 1))

J = cost(theta, X, y)
print(J)
