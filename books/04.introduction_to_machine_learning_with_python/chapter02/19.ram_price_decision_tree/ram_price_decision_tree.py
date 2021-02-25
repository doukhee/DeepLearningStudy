#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# csv 파일 읽어오기
ram_price = pd.read_csv("../../data/ram_price.csv")
# 읽어온 데이터 확인을 위한 출력
print(ram_price)
# semilogy는 y축에 대해 로그 스케일을 사용하여 데이터를 그래프 그리는 함수
plt.semilogy(ram_price.date, ram_price.price)
# 라벨 설정
plt.xlabel("year")
plt.ylabel("price ($/Mbyte_")
# 그래프 보이기
plt.show()
#2000년 이전을 훈련데이터로 설정, 2000년 이ㅣ후를 테스트 데이터로 설정
data_train = ram_price[ram_price.date < 2000]
data_test = ram_price[ram_price.date >= 2000]
# 가격 예측을 위해 날짜 특성을 이용
X_train = data_train.date[:, np.newaxis]
# 데이터와 타겟의 관계를 간단하게 만들기 위해 로그 스케일로 바꿉니다
y_train = np.log(data_train.price)
# 결정 트리로 학습
tree = DecisionTreeRegressor().fit(X_train, y_train)
# 선형회귀모델로 학습
linear_reg = LinearRegression().fit(X_train, y_train)
# 행렬의 차원을 확장 시키기
X_all = ram_price.date[:, np.newaxis]
# 예측 하기
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)
# 지수함수로 변경
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)
# 그래프 그리기
plt.semilogy(data_train.date, data_train.price, label="train data")
plt.semilogy(data_test.date, data_test.price, label="test data")
plt.semilogy(ram_price.date, price_tree, label="tree predict")
plt.semilogy(ram_price.date, price_lr, label="linear predict")
# 범주 추가
plt.legend()
# 그래프 보이기
plt.show()