# coding: utf-8

import pandas as pd
import numpy as np


X = pd.read_table('./ex4Data/ex4x.dat', sep='   ')
Y = pd.read_table('./ex4Data/ex4y.dat', sep='   ')

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