#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
# 웨이브 데이터 생성 및 샘플링을 40번만 하는 데이터 생성
X, y = mglearn.datasets.make_wave(n_samples=40)
# 테스트 데이터 및 학습 데이터로 랜덤으로 분류
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 화면 생성 및 분활
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# -3과 3 사이에 1000개의 데이터 포인트를 만든다
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
# 
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 1, 3, 9 이웃을 사용하여 예측
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    # 학습 하기
    reg.fit(X_train, y_train)
    # 예측 값을 그래프 그리기
    ax.plot(line, reg.predict(line))
    # 예측 값을 그래프 그리기
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    # 예측 값을 그래프 그리기
    ax.plot(X_test, y_test, "v", c=mglearn.cm2(1), markersize=8)
    # 그래프 타이틀 생성
    ax.set_title("{} neighbor train score : {:.2f} test score : {:.2}".format(n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
    # x 라벨 값 설정
    ax.set_xlabel("property")
    # y 라벨 값 설정
    ax.set_ylabel("targe")
# 범주 설정
axes[0].legend(["model predict", "train data/target", "test data/target"], loc="best")
# 그래프 보이기
plt.show()