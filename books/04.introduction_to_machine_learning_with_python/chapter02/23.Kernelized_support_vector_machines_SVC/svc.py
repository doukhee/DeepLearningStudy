#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
# 분류용 가상 데이터 생성
X, y = make_blobs(centers=4, random_state=8)
y = y % 2
# 선점도 그리기
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# 라벨 설정
plt.xlabel("property 0")
plt.ylabel("property 1")
# 그래프 표출
plt.show()
# SVC 분류 하고, 학습하기
Linear_svm = LinearSVC().fit(X, y)
# 2차원 선점도 그리기
mglearn.plots.plot_2d_separator(Linear_svm, X)
# 선점도 그리기
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# 라벨 추가
plt.xlabel("property 0")
plt.ylabel("property 1")
# 그래프 보이기
plt.show()