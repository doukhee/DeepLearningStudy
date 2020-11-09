#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2])
print("X shape {}".format(X.shape))

W = np.array([[1,3,5], [2,4,6]])
print("W is {}".format(W))
print("W shape {}".format(W.shape))

Y = np.dot(X, W)
print("Y is dot(X, W) : {}".format(Y))

