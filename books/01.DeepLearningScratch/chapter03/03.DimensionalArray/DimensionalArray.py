#coding; utf-8
import numpy as np

A = np.array([1, 2, 3, 4])
print(A)
print("Dimensional : {}".format(np.ndim(A)))

print("Array Shape : {}".format(A.shape))

B = np.array([[1,2], [3, 4], [5, 6]])

print(B)

print("Dimensional : {}".format(np.ndim(B)))

print("Array Shape : {}".format(B.shape))