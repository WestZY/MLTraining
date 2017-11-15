# coding: utf-8
import numpy as np
from my_util import *

class MLP:
    def __init__(self, shape, r):
        #load data
        data = np.loadtxt('./nn_data/hw4_nnet_train.dat')
        #split data
        self.X = np.column_stack((np.ones(len(data)), data[:, :-1]))
        self.Y = data[:, -1].reshape(len(data), 1)
        self.shape = shape
        #random thetas
        self.thetas = [uniform_rand(r, (shape[i] + 1, shape[i + 1])) for i in range(len(shape) - 1)]
        print self.thetas

    def forward_propagation(self):
        s1 = tanh(np.dot(self.X, self.thetas[0]))
        s2 = tanh(np.dot(np.column_stack((np.ones(len(s1)), s1)) ,self.thetas[1]))
        return s1, s2

    def back_propagation(self, s1, s2):
        delta_3 = np.power(s2 - self.Y, 2) / 2
        print delta_3.shape, self.thetas[1].shape, deri_tanh(s2).shape
        delta_2 = np.dot(self.thetas[1] * delta_3.T, deri_tanh(s2))
        print delta_2
        print delta_2.shape, self.thetas[0].shape, deri_tanh(s1).shape
        delta_1 = np.dot(self.thetas[0] * delta_2[1:, ].T, deri_tanh(s1))
        print delta_1



    def run(self):
        s1, s2 = self.forward_propagation()
        self.back_propagation(s1, s2)

if __name__ == "__main__":
    nn = MLP([2, 3, 1], 0.01)
    nn.run()

