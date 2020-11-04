#coding: utf-8
import numpy as np

A = np.array([[1,2],[3,4]])
B = np.array([10,20])

print(A*B)

X = np.array([[51,55],[14,19],[0,4]])

print(X)

print(X[0][1])

for row in X:
    print(row)

# 일차원 배열로 변경하는 함수 평탄화 작업
X = X.flatten()
print(X)

print(X[np.array([0, 2, 4])])

print(X > 15)

print(X[X>15])