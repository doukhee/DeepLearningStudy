#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 유방암 데이터 가져오기
cancer = load_breast_cancer()
# 데이터를 테스트 데이터와 학습 데이터로 랜덤하게 나누기
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# 로지스틱 회귀(분류)로 학습 데이터 학습
logreg = LogisticRegression().fit(X_train, y_train)
# 학습 결과 출력
print("Training sets score : {:.3f}".format(logreg.score(X_train, y_train)))
# 테스트 결과 출력
print("Test sets score : {:.3f}\r\n".format(logreg.score(X_test, y_test)))
# 로지스틱 회귀(분류)로 학습 데이터 학습 제약을 풀어서 실행 개개의 데이터 포인트를 정확하게 분류
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
# 학습 결과 출력
print("Training sets score : {:.3f}".format(logreg100.score(X_train, y_train)))
# 테스트 결과 출력
print("Test sets score : {:.3f}\r\n".format(logreg100.score(X_test, y_test)))
# 로지스틱 회귀(분류)로 학습 데이터 학습 제약을 더 주어서 실행 개개의 데이터 포인트를 대략적으로 분류
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
# 학습 결과 출력
print("Training sets score : {:.3f}".format(logreg001.score(X_train, y_train)))
# 테스트 결과 출력
print("Test sets score : {:.3f}\r\n".format(logreg001.score(X_test, y_test)))
# 계수의 값을 그리기
plt.plot(logreg.coef_.T, "o", label="C=1")
# 계수의 값을 그리기
plt.plot(logreg100.coef_.T, "^", label='C=100')
# 계수의 값을 그리기
plt.plot(logreg001.coef_.T, "v", label="C=0.001")
# x축의 눈금을 유방암 데이터의 수 만큼 만들고, 값을 feature_names로 설정 및 90도 회전
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
# 수평선 생성
plt.hlines(0, 0, cancer.data.shape[1])
# y의 값을 -5 부터 5까지로 제한
plt.ylim(-5, 5)
# x 라벨 값 정의
plt.xlabel("property")
# y 라벨 값 정의
plt.ylabel("weight size")
# 범주 표시
plt.legend()
# 그래프 보여주기
plt.show()