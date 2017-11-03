# coding: utf-8

import pandas as pd
import numpy as np
from gradientDescentMulti import gradient_descent, cost
from sklearn.model_selection import train_test_split

#load data
X = pd.read_table('./ex2Data/ex2x.dat', sep='   ')
Y = pd.read_table('./ex2Data/ex2y.dat', sep='   ')
X1 = np.column_stack((np.ones(X.shape[0]), X))
X2 = np.column_stack((np.ones(X.shape[0]), X, np.square(X)))
X3 = np.column_stack((np.ones(X.shape[0]), X, np.square(X), np.power(X, 3)))
#split data
X1_train, X1_test, Y1_train, Y1_test = \
    train_test_split(X1, Y, test_size=0.3, random_state=0)
X2_train, X2_test, Y2_train, Y2_test = \
    train_test_split(X2, Y, test_size=0.3, random_state=0)
#X3_train, X3_test, Y3_train, Y3_test = \
#    train_test_split(X3, Y, test_size=0.3, random_state=0)
#training
alpha = 0.001
lamda = 1
theta1 = gradient_descent(X1_train, Y1_train, alpha, lamda, 1000)
print theta1
print cost(X1_test, Y1_test, theta1)
theta2 = gradient_descent(X2_train, Y2_train, alpha, lamda, 1000)
print theta2
print cost(X2_test, Y2_test, theta2)
#theta3 = gradient_descent(X3_train, Y3_train, alpha, lamda, 1000)
#print theta3
#print cost(X3_test, Y3_test, theta3)





