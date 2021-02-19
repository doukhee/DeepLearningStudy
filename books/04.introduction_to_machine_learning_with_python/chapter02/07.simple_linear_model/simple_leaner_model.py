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
from sklearn.linear_model import LinearRegression
# 웨이브 데이터 만들기
X, y = mglearn.datasets.make_wave(n_samples=60)
# 학습 및 테스트를 위한 값 분류
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# 선형 회귀(최소 제곱 법)으로 핛습
lr = LinearRegression().fit(X_train, y_train)
# 가중치 출력
print("lr.coef_: {}".format(lr.coef_))
# 절편 출력
print("lr.iintercept_ : {}".format(lr.intercept_))
# 학습 데이터의 결과 출력
print("Training set score : {:.2f}".format(lr.score(X_train, y_train)))
# 테스트의 결과 출력
print("Test set score : {:.2}".format(lr.score(X_test, y_test)))