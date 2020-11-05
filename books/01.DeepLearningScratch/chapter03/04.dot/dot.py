# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

A = np.array([[1,2],[3,4]])

print("A array shape : {}".format(A.shape))

B = np.array([[5, 6], [7, 8]])
print("B array shape : {}".format(B.shape))

print("A dot B is {}".format(np.dot(A, B)))