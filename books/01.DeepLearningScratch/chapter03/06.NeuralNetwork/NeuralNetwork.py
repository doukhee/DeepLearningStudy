#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 입력값
X = np.array([1.0,0.5])
# 가중치
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
# 편향
B1 = np.array([0.1,0.2,0.3])

print("가중치의 모양 : {}".format(W1.shape))
print("입력 의 모양 : {}".format(X.shape))
print("편향의 모양 : {}".format(B1.shape))

# 신경망 계산
A1 = np.dot(X, W1) + B1

print("NeuralNetwork Value : {}".format(A1))