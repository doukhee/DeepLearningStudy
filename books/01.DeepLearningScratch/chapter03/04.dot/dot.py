# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

A = np.array([[1,2],[3,4]])

print("A array shape : {}".format(A.shape))

B = np.array([[5, 6], [7, 8]])
print("B array shape : {}".format(B.shape))

print("A dot B is {}".format(np.dot(A, B)))

A = np.array([[1, 2], [3, 4], [5, 6]])

print("A shape {}".format(A.shape))

B = np.array([7, 8])

print("B shape {}".format(B.shape))

print("A dot B {}".format(np.dot(A, B)))

X = np.array([1,2])

print("X shape {}".format(X.shape))

W = np.array([[1, 3, 5], [2, 4, 6]])

print("W shape {}".format(W.shape))

Y = np.dot(W, X)

print("W dot X : {}".format(np.dot(W, X)))
