#!/usr/bin/env python3
#-*- coding: utf-8 -*-
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
# 보스턴 주택 가격을 불러온다
boston = load_boston()
# 데이터의 모양 출력
print("data shape : {}".format(boston.data.shape))
# 특성공학을 사용하여 데이터 특성 늘리기
X, y = mglearn.datasets.load_extended_boston()
# 데이터 모양 출력
print("X.shape: {}".format(X.shape))
# K-NN 알고리즘을 이용하여 가장 가까운 훈련 데이터 포인프 하나를 최근접 이웃으로 찾아 예측
mglearn.plots.plot_knn_classification(n_neighbors=1)
# 차트 출력
plt.show()
# K-NN 알고리즘을 이용하여 가장 가까운 훈련 데이터 포인프 세개를 최근접 이웃으로 찾아 예측
mglearn.plots.plot_knn_classification(n_neighbors=3)
# 차트 출력
plt.show()
# 기존 데이터 셋 초기화
X, y = mglearn.datasets.make_forge()
# 트레이닝 및 테스트 데이터 나누기 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# K-NN 알고리즘을 이용하여 가장 가까운 훈련 데이터 포인프 세개를 최근접 이웃으로 찾아 예측
clf = KNeighborsClassifier(n_neighbors=3)
# 학습 시키기?
clf.fit(X_train, y_train)
# 확인을 위한 한 줄 띄기
print("")
# 예측 값 출력
print("Test set Prediction : {}".format(clf.predict(X_test)))
# 정확도 출력
print("Test accuracy : {:.2f}".format(clf.score(X_test, y_test)))
# 화면 분활하기
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
# 
for n_neighbors, ax in zip([1, 3, 9], axes):
    # fit 메서드는 slef 객체를 반환한다
    # 그래서 객체 생ㅅ어과 fit메서드를 한주롤 쓸 수 있다.
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    # 2D의 산점도를 그린다
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    # 산점도를 그린다
    mglearn.discrete_scatter(X[:,0], X[:, 1], y, ax=ax)
    # 그래프의 타이틀 설정
    ax.set_title("{} neighbor".format(n_neighbors))
    # 그래프의 x축 타이틀 설정
    ax.set_xlabel("property 0")
    # 그래프의 y축 타이틀 설정
    ax.set_ylabel("property 1")
# 그래프의 범주 및 위치를 정의ㄴ
axes[0].legend(loc=3)
# 그래프 보여주기
plt.show()