#!/usr/bin/env python3
#-*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
# 붓꽃의 데이터 가져오기
iris_dataset = load_iris()
# 아이리스의 키 값 출력
print("iris_dataset key : \r\n{}".format(iris_dataset.keys()))
# 데이터 셋에 대한 것이 담겨있는 열을 선택 후 193번째 값만 표출
print(iris_dataset['DESCR'][:193] + "\r\n")
# 예측하려는 붓꽃 품종의 이름 출력
print("target name : \r\n{}".format(iris_dataset['target_names']))
# 각 특성을 설명하는 문자열 리스트 출력
print('feature name : \r\n{}'.format(iris_dataset['feature_names']))
# 데이터의 타입 출력
print("data type : \r\n{}".format(type(iris_dataset['data'])))
# 데이터 행의 개개의 꽃이 되며 각 꽃에서 구한 네개의 측정치 출력
print("data size : {}".format(iris_dataset['data'].shape))
# 데이터 행의 5줄 보여주기
print("data first 5 rows : \r\n{}".format(iris_dataset['data'][:5]))
# 데이터 배열의 타입을 출력
print("target type : {}".format(type(iris_dataset['target'])))
# 데이터 배열의 크기 출력
print("target size : {}".format(iris_dataset['target'].shape))
# 타겟 데이터 출력
print("target \r\n:{}".format(iris_dataset['target']))
# 타겟 데이터의 이름 출력
print("target names : {}".format(iris_dataset['target_names']))
# 테스트 데이터와 학습 데이터로 나누기
X_train,X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
# X_train 크기 출력
print("X_train size : {}".format(X_train.shape))
# y_train 크기 출력
print("y_train size : {}".format(y_train.shape))
# X_test 크기 출력
print("X_test size : {}".format(X_test.shape))
# y_test 크기 출력
print("y_train size : {}".format(y_test.shape))
# pandas의 DataFrame으로 변경 열의 이름은 iris_dataset.feature_names를 사용한다
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 산점도 차트를 그리기
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
# 화면 보여주기
#plt.show()
# k 근접을 사용하기 위한 함수 n_neighbors는 근접 값을 몇개 찾을지 설정하는 것
knn = KNeighborsClassifier(n_neighbors=1)
# 군집화 진행 후 값 출력
print("knn clustering : {}".format(knn))
# 훈련 데이터 세트에서 k- 최근 접 이웃 분류기를 피팅합니다.
knn.fit(X_train, y_train)
# 피팅 값 출력
print("knn fit function : {}".format(knn.fit(X_train, y_train)))
# 새로운 배열 생성
X_new = np.array([[5, 2.9, 1, 0.2]])
# 새로운 배열의 모양 출력
print("X_new.shape : {}".format(X_new.shape))
# 새로운 값을 예측 하기
prediction = knn.predict(X_new)
# 예측 값 출력
print("Prediction : {}".format(prediction))
# 예측한 모델 값을 출력
print("Prediction target name : {}".format(iris_dataset['target_names'][prediction]))
# 예측 하기
y_pred = knn.predict(X_test)
# 예측 값 출력
print("Test set prediction : \r\n{}".format(y_pred))
# 예측 값의 정확도 출력 np.mean()은 지정된 축을 따라 산술 평균 을 계산 값
print("Test set accuracy : {:.2f}".format(np.mean(y_pred == y_test)))
# 예측 값의 정확도 출력 knn.score()는 결과 값의 정확도 출력 하는 함수
print("Test sett accuracy : {:.2f}".format(knn.score(X_test, y_test)))
# 테스트 데이터 만들기
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
# K 근접을 사용하기 위한 함수
knn = KNeighborsClassifier(n_neighbors=1)
# 피팅 하기
knn.fit(X_train, y_train)
# 정확도를 구하는 것
print("Test Set accuracy : {:.2f}".format(knn.score(X_test, y_test)))