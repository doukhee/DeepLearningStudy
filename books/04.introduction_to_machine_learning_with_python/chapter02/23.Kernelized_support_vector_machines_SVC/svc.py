#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
# 3차원 그래프 그리기 위한 3차원 모듈 추가
from mpl_toolkits.mplot3d import Axes3D, axes3d
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
# 배열을 가로로 합치기
X_new = np.hstack([X, X[:, 1:] ** 2])
# 화면 생성
figure = plt.figure()
# 3차원 그래프 그리기
ax = Axes3D(figure, elev=-152, azim=-26)
# 마스크 값과 y값 설정
mask = y == 0
# 선점도 그리기
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60, edgecolor='k')
# 선점도 그리기
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60, edgecolor='k')
# 라벨 설정
ax.set_xlabel("property 0")
ax.set_ylabel("property 1")
ax.set_zlabel("property 1 ** 2")
# 그래프 표출
plt.show()
# 선형 SVC로 학습 하기
linear_svm_3d = LinearSVC().fit(X_new, y)
# 절편 및 절편 구하기
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
# 화면 생성
figure = plt.figure()
# 3차원 그래프 그리기
ax = Axes3D(figure, elev=-152, azim=-26)
# 일차원 그래프 만들기
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:,1].min() - 2, X_new[:, 1].max() + 2, 50)
# 그리드 생성
XX, YY = np.meshgrid(xx, yy)
# Z 축 데이터 생성
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
# 표면 플롯 만들기
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
# 선점도 생성
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60, edgecolor='k')

# 라벨 설정
ax.set_xlabel("property 0")
ax.set_ylabel("property 1")
ax.set_zlabel("property 1 ** 2")
# 그래프 표출
plt.show()

ZZ = YY ** 2

dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])

plt.contourf(XX, YY, dec.reshape(), levels=[dec.min(), 0, dec.max()], cmap=mglearn.cm2, alpha=0.5)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

plt.xlabel("property 0")
plt.ylabel("property 1")

plt.show()