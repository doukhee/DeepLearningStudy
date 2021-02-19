#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# 보스턴 데이터를 확장해서 가져오기
X, y = mglearn.datasets.load_extended_boston()
# 학습 데이터 테스트 데이터로 랜덤하게 분류
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 선형 회귀로 학습
lr = LinearRegression().fit(X_train, y_train)
# 릿지회귀로 학습 성능대비 단순화하기 위한 alpha값 설정 계수를 0에 더 가깝게 만들면 훈련 성능이 나빠지지만, 일반화에는 좋다
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
# 릿지회귀로 학습 성능대비 단순화하기 위한 alpha값 설정 계수를 0에 더 가깝게 만들면 훈련 성능이 나빠지지만, 일반화에는 좋다
ridge = Ridge().fit(X_train, y_train)
# 릿지회귀로 학습 성능대비 단순화하기 위한 alpha값 설정 계수를 0에 더 가깝게 만들면 훈련 성능이 나빠지지만, 일반화에는 좋다
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
# 학습 성능 출력
print("Train set score : {:.2}".format(ridge10.score(X_train, y_train)))
# 테스트 성능 출력
print("Test set score : {:.2}".format(ridge10.score(X_test, y_test)))
# 가중치를 그리기 알파를 10으로 했을 때
plt.plot(ridge10.coef_, "^", label="Ridge alpha=10")
# 가중치를 그리기 알파를 1으로 했을 때
plt.plot(ridge.coef_, "s", label="Ridge alpha=1")
# 가중치를 그리기 알파를 0.1으로 했을 때
plt.plot(ridge01.coef_, "v", label="Ridge alpha=0.1")
# 선형 회귀의 가중치 그리기
plt.plot(lr.coef_, 'o', label="LinearRegression")
# x축의 라벨 설정
plt.xlabel("weight list")
# y축의 라벨 설정
plt.ylabel("weight size")
# 0 0으로 수평의 라인 그리기
plt.hlines(0, 0, len(lr.coef_))
# y축 값 제한
plt.ylim(-25, 25)
# 범례 표현 설정
plt.legend()
# 차트 그리기
plt.show()
# 학습 곡선 그리기
mglearn.plots.plot_ridge_n_samples()
# 차트 그리기
plt.show()