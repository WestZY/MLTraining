# coding: utf-8

import pandas as pd
import numpy as np
from numpy import where
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import math
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, x_path, y_path, sep='   '):
        self.X = pd.read_table(x_path, sep=sep)
        self.Y = pd.read_table(y_path, sep=sep)
        self.X1 = np.column_stack((np.ones(self.X.shape[0]), self.X))
        poly = PolynomialFeatures(2)
        self.X2 = poly.fit_transform(self.X)

        #split data
        self.X1_train, self.X1_test, self.Y1_train, self.Y1_test \
            = train_test_split(self.X1, self.Y, test_size=0.3, random_state=0)
        self.X2_train, self.X2_test, self.Y2_train, self.Y2_test \
            = train_test_split(self.X2, self.Y, test_size=0.3, random_state=0)

    def sigmod(self, x):
        return np.longfloat(1 / (1 + np.exp(-x)))

    def gradient_descent(self, x, y, lamda=1, num_iters=1000):
        m, n = x.shape
        theta_g = np.zeros((n, 1))
        for i in range(num_iters):
            for j in range(n):
                theta_temp = theta_g.copy()
                x_temp = x[:, j].reshape(m, 1)
                if j == 0:
                    #print (self.sigmod(np.dot(x, theta_g)) - y)*x_temp
                    theta_temp[j] = np.sum((self.sigmod(np.dot(x, theta_g)) - y) * x_temp) / m
                else:
                    theta_temp[j] = \
                        np.sum((self.sigmod(np.dot(x, theta_g)) - y) * x_temp) / m  \
                        - lamda * theta_temp[j] / m
            theta_g = theta_temp.copy()
        return theta_g

    def run(self):
        # training
        lamda = 10
        number_iter = 100
        theta = self.gradient_descent(self.X1_train, self.Y1_train, lamda, number_iter)
        print theta

lr = LogisticRegression('./ex4Data/ex4x.dat', './ex4Data/ex4y.dat')
lr.run()
