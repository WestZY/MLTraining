# coding: utf-8
import numpy as np
import math


def gradient_descent(x, y, alpha=0.01, lamda=0.001, num_iters=1000):
    m, n = x.shape
    theta = np.zeros(n).reshape(n, 1)

    for i in range(num_iters):
        theta_tmp = theta.copy()

        for j in range(n):
            x_1 = x[:, j].reshape(m, 1)
            temp = theta[j] * (1 - alpha * lamda / m) - \
                       alpha * np.sum((np.dot(x, theta) - y) * x_1) / m
            if math.isinf(temp) or math.isnan(temp):
                return theta
            else:
                theta_tmp[j] = temp
        theta = theta_tmp

    return theta


def cost(x, y, theta):
    m = x.shape[0]
    return round(np.sum(np.power(np.dot(x, theta) - y, 2)) / m / 2, 10)


