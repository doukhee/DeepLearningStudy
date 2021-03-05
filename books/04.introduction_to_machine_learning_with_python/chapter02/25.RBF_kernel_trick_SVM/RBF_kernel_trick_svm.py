#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
import mglearn
from IPython.display import display
# 학습할 데이터 불러오기
cancer = load_breast_cancer()
# 학습할 데이터 분류하기
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# 학습 모델 생성
svc = SVC()
# SVC로 훈련 시킨다
svc.fit(X_train, y_train)
# 학습률 및 예측률 표시
print("train set accuracy score : {:.2f}".format(svc.score(X_train, y_train)))
print("test set accuracy score : {:.2f}\r\n".format(svc.score(X_test, y_test)))
#  이 그래프는 가공하지 않은 자료 그대로를 이용하여 그린 것이 아니라, 자료로부터 얻어낸 통계량인 5가지 요약 수치(다섯 숫자 요약, five-number summary)를 가지고 그린다.
# 이 때 5가지 요약 수치란 최솟값, 제 1사분위({\displaystyle Q_{1}}Q_{1}), 제 2사분위({\displaystyle Q_{2}}Q_{2}), 제 3사분위({\displaystyle Q_{3}}{\displaystyle Q_{3}}), 최댓값을 일컫는 말이다.
# 히스토그램과는 다르게 집단이 여러개인 경우에도 한 공간에 수월하게 나타낼수 있다.
plt.boxplot(X_train, manage_ticks=False)
# y축의 스케일 지정 
# yscale() -
# linear - 축의 스케일을 선형으로 만든다.
# log - 축의 스케일을 로그로 만든다.
# symlog - 축의 스케일을 시메트릭로그 스케일로 만든다. 로그 스케일에서 불가능한 음수 영역을 표현 해주는 것이 가능해진다.
# logit - 축의 스케일을 로짓스 스케일로 만든다. 0과 1사이에 포함되는 데이터만으로 구성된다.
plt.yscale("symlog")
# 라벨 표시
plt.xlabel("property list")
plt.ylabel("property size")
# 그래프 보여주기
plt.show()
# 훈련 세트에서 특성별 최솟값 계산
min_on_training = X_train.min(axis=0)
# 훈련 세트에서 특성별 (최댓값 - 최솟값) 범위 계산
range_on_training = (X_train - min_on_training).max(axis=0)
# 훈련 데이터에 최솟값을 빼고, 범위로 나누면 각 특성에 대해 최솟값은 0, 최대 값은 1입니다.
X_train_scaled = (X_train - min_on_training) / range_on_training
# 특성별 최솟 값 및 최대값 출력
print("property min value : \r\n{}".format(X_train_scaled.min(axis=0)))
print("property max value : \r\n{}".format(X_train_scaled.max(axis=0)))

# 테스트 세트에도 같은 작업을 적용하지만,
# 훈련세트에서 계산한 최솟값과 범위를 사용합니다.
X_test_scaled = (X_test - min_on_training) / range_on_training
# 분류기 생성
svc = SVC()
# 학습 시키기
svc.fit(X_train_scaled, y_train)
# 정확도 출력
print("train set accuracy : {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("test set accuracy : {:.3f}".format(svc.score(X_test_scaled, y_test)))
# 분류기 생성
svc = SVC(C=1000)
# 학습 시키기
svc.fit(X_train_scaled, y_train)
# 정확도 출력
print("train set accuracy : {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("test set accuracy : {:.3f}".format(svc.score(X_test_scaled, y_test)))