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

# 유방암 데이터 가져오기
cancer = load_breast_cancer()
# 데이터를 학습 데이터와 테스트 데이터로 분활
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
# 학습 값의 정확도를 담을 배열
training_accuracy = []
# 테스트 값의 정확도를 담을 배열
test_accuracy = []
# knn의 변수를 변경을 해줄 값들의 배열 선언
neighbors_settings = range(1, 11)
# knn의 변수를 변경하면서 확인하기 위한 반복문
for n_neighbors in neighbors_settings:
    # knn 분류 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    # 학습 데이터로 학습
    clf.fit(X_train, y_train)
    # 학습 정확도 배열에 추가
    training_accuracy.append(clf.score(X_train, y_train))
    # 테스트 정확도 배열에 추가
    test_accuracy.append(clf.score(X_test, y_test))
# 차트 그리기
plt.plot(neighbors_settings, training_accuracy, label="Train accuracy")
# 차트 그리기
plt.plot(neighbors_settings, test_accuracy, label="Test accuracy")
# y축의 타이틀 설정
plt.ylabel("accuracy")
# x축의 타이틀 설정
plt.xlabel("n_neighbors")
# 차트 라벨 설정
plt.legend()
# 차트 보여주기
plt.show()
# knn을 이용한 회귀 분석 
mglearn.plots.plot_knn_regression(n_neighbors=1)
# 차트 보여주기
plt.show()
# knn을 이용한 회귀 분석 
mglearn.plots.plot_knn_regression(n_neighbors=3)
# 차트 보여주기
plt.show()

# 웨이브 데이터를 40개 샘플링 으로 만들기
X, y = mglearn.datasets.make_wave(n_samples=40)
# 테스트 데이터와 학습 데이터를 랜덤으로 나누기
X_train, X_test, y_train, y_train = train_test_split(X, y, random_state=0)
# 회귀 분류로 변경
reg = KNeighborsRegressor(n_neighbors=1)
# 학습 시키기
print(X_train.shape)
print(y_train.shape)
# train 데이터의 행렬이 맞지 않아 서 학습이 안된다s
#reg.fit(X_train, y_train)
# 예측 값 출력
#print("Test set prediction : \r\n{}".format(reg.predict(X_test)))
# 테스트 결과 예측 값 출력
#print("test set R^2 : {:.2f}".format(reg.score(X_test, y_test)))
