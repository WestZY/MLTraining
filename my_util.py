# coding: utf-8
import numpy as np


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def uniform_rand(r, shape):
    return 2 * r * np.random.random(shape) - r

def deri_tanh(x):
    return 1 - np.power(tanh(x), 2)

