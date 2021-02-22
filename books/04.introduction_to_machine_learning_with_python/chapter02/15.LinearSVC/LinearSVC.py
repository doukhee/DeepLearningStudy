#/!usr/bin/env python3
#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
# support vector machine algorism
from sklearn.svm import LinearSVC
# 가상의 데이터 생성
X, y = make_blobs(random_state=42)
#선형 지원 벡터 분류.
# 매개 변수 kernel = 'linear'가있는 SVC와 유사하지만 libsvm이 아닌
# liblinear 측면에서 구현되었으므로 페널티 및 손실 함수를 선택할 때
# 더 많은 유연성이 있으며 많은 수의 샘플로 더 잘 확장되어야합니다.
Linear_svm = LinearSVC().fit(X, y)
# 절편과 가중치 모양 확인을 위한 출력
print("weight array size : ", Linear_svm.coef_.shape)
print("intercept array size : ", Linear_svm.intercept_.shape)
# 섡점도 그리기
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# -15부터 15까지 균일한 간격으로 시퀀스 생성
line = np.linspace(-15, 15)
# 반복문으로 그래프 그리기
for coef, intercept, color in zip(Linear_svm.coef_, Linear_svm.intercept_, mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
# 그래프 값 제한
plt.ylim(-10, 15)
plt.xlim(-10, 8)
# 라벨 설정
plt.xlabel("property 0")
plt.ylabel("property 1")
# 그래프 범주 설정 및 그리기
plt.legend(['class 0', 'class 1', 'class 2', 'class 0 boundary', 'class 1 boundary', 'class 2 boundary'])
# 그래프 그리기
plt.show()
# 2차원 분류
mglearn.plots.plot_2d_classification(Linear_svm, X, fill=True, alpha=.7)
# 선점도 그리기
mglearn.discrete_scatter(X[:,0], X[:,1], y)
# -15부터 15까지 균일한 간격으로 시퀀스 생성
line = np.linspace(-15, 15)
# 반복문으로 그래프 그리기
for coef, intercept, color in zip(Linear_svm.coef_, Linear_svm.intercept_, mglearn.cm3.colors):
    # 그래프 그리기
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
# 범주 설정
plt.legend(['class 0', 'class 1', 'class 2', 'class 0 boundary', 'class 1 boundary', 'class 2 boundary'], loc=(1.01, 0.3))
# 라벨 설정
plt.xlabel('property 0')
plt.ylabel('property 1')
# 그래프 그리기
plt.show()