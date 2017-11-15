# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt

class LinearRegression:
    # load data
    def __init__(self, x_path, y_path, sep='   '):
        self.X = pd.read_table(x_path, sep=sep)
        self.Y = pd.read_table(y_path, sep=sep)
        self.X1 = np.column_stack((np.ones(self.X.shape[0]), self.X))
        self.X2 = np.column_stack((np.ones(self.X.shape[0]), \
                                   self.X, np.square(self.X)))
        self.X1_train, self.X1_test, self.Y1_train, self.Y1_test = \
            train_test_split(self.X1, self.Y, test_size=0.3, random_state=0)
        self.X2_train, self.X2_test, self.Y2_train, self.Y2_test = \
            train_test_split(self.X2, self.Y, test_size=0.3, random_state=0)



    def gradient_descent(self, x, y, alpha=0.01, lamda=1, num_iters=1000):
        m, n = x.shape
        theta = np.zeros(n).reshape(n, 1)

        for i in range(num_iters):
            theta_tmp = theta.copy()

            for j in range(n):
                x_1 = x[:, j].reshape(m, 1)
                #theta0
                if j == 0:
                    temp = theta[j] \
                           - alpha * np.sum((np.dot(x, theta) - y) * x_1) / m
                #theta j
                else:
                    temp = theta[j] * (1 - alpha * lamda / m) - \
                           alpha * np.sum((np.dot(x, theta) - y) * x_1) / m
                if math.isinf(temp) or math.isnan(temp):
                    return theta
                else:
                    theta_tmp[j] = temp
            theta = theta_tmp

        return theta

    def cost(self, x, y, theta):
        m = x.shape[0]
        return round(np.sum(np.power(np.dot(x, theta) - y, 2)) / m / 2, 10)

    def run(self):
        # training
        alpha = 0.001
        lamda = 1
        number_iter = 500
        x = []
        y = []
        for i in range(50, number_iter, 50):
            theta = self.gradient_descent(self.X1_train, self.Y1_train, alpha, lamda, i)
            score = self.cost(self.X1_test, self.Y1_test, theta)
            x.append(i)
            y.append(score)
            print i, round(score, 5)

        plt.plot(x, y)
        plt.axis([0, number_iter, 0, 0.1])
        plt.show()
'''
        theta2 = self.gradient_descent(self.X2_train, self.Y2_train, alpha, lamda, number_iter)
        print self.cost(self.X2_test, self.Y2_test, theta2)

        alpha = 0.003
        lamda = 1
        number_iter = 1000
        print alpha, lamda
        theta1 = self.gradient_descent(self.X1_train, self.Y1_train, alpha, lamda, number_iter)
        print self.cost(self.X1_test, self.Y1_test, theta1)

        theta2 = self.gradient_descent(self.X2_train, self.Y2_train, alpha, lamda, number_iter)
        print self.cost(self.X2_test, self.Y2_test, theta2)
'''
lr = LinearRegression('./ex2Data/ex2x.dat', './ex2Data/ex2y.dat')
lr.run()






