#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# 보스턴 데이터를 확장해서 가져오기
X, y = mglearn.datasets.load_extended_boston()
# 학습 데이터 테스트 데이터로 랜덤하게 분류
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 릿지회귀로 학습 성능대비 단순화하기 위한 alpha값 설정 계수를 0에 더 가깝게 만들면 훈련 성능이 나빠지지만, 일반화에는 좋다
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
# Lasso L1 규제로 학습하기
lasso = Lasso().fit(X_train, y_train)
# 트레이닝 정확도 출력
print("Train set Score : {:.2f}".format(lasso.score(X_train, y_train)))
# 테스트 정확도 출력
print("Test set Score : {:.2f}".format(lasso.score(X_test, y_test)))
# 사용한 특성 수 출력
print("Using property number : {}".format(np.sum(lasso.coef_ != 0)))
# "max_iterr" 기본 값을 증가 시키지 않으면, max_iter 값을 늘리라는 경고 발생
# 알파 값으로 규제 설정 하여 Lasso 로 학습
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
# 출력 확인을 위한 줄 변경
print("")
# 트레이닝 정확도 출력
print("Train set score : {:.2f}".format(lasso001.score(X_train, y_train)))
# 테스트 정확도 출력
print("Test set score : {:.2}".format(lasso001.score(X_test, y_test)))
# 사용한 특성 수 출력
print("using Property number : {}\r\n".format(np.sum(lasso001.coef_ != 0)))

lasso0001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
# 트레이닝 정확도 출력
print("Train set score : {:.2f}".format(lasso0001.score(X_train, y_train)))
# 테스트 정확도 출력
print("Test set score : {:.2}".format(lasso0001.score(X_test, y_test)))
# 사용한 특성 수 출력
print("using Property number : {}\r\n".format(np.sum(lasso0001.coef_ != 0)))

# Lasso alpha=1일때, 가중치 차트 그리기
plt.plot(lasso.coef_, "s", label="Lasso alpha=1")
# Lasso alpha=0.01일때, 가중치 차트 그리기
plt.plot(lasso001.coef_, "^", label="Lasso alpha=0.01")
# Lasso alpha=0.001일때, 가중치 차트 그리기
plt.plot(lasso0001.coef_, "v", label="Lasso alpha=0.001")
# Ridge alpha=0.1일때, 카중치 차트 그리기
plt.plot(ridge01.coef_, "o", label="RRidge alpha=0.1")
# 범주 위치 및 생성
plt.legend(ncol=2, loc=(0, 1.05))
# y 값 제한
plt.ylim(-25, 25)
# x라벨 설정
plt.xlabel("weight list")
# y라벨 설정
plt.ylabel("weight size")
# 차트 보여주기
plt.show()
