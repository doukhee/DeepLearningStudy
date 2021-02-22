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

# C의 값을 변경을 하면서 차트를 그리기 위한 반복문
for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    # 로지스틱 회귀(분류)로 학습 데이터 학습 L1
    lr_l1 = LogisticRegression(C=C, penalty="l1", solver='liblinear').fit(X_train, y_train)
    # 정확도 및 C의 값 출력
    print("C={:.3f} L1 Logistic Regression train accuracy : {:.2f}".format(C, lr_l1.score(X_train, y_train)))
    # 정확도 및 C의 값 출력
    print("C = {:.3f} L1 Logistic Regression Test accuracy : {:.2f}".format(C, lr_l1.score(X_test, y_test)))
    # 계수의 값을 그릭
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
# x의 눈금 표시 및 값 정의 와 90도 회전
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
# 수평선 그리기
plt.hlines(0, 0, cancer.data.shape[1])
# x라벨 추가
plt.xlabel("property")
# y라벨 추가
plt.ylabel("weight size")
# y 값 -5부터 5까지로 제한
plt.ylim(-5, 5)
# 범주 위치 설정 미ㅣㅊ 범주 생성
plt.legend(loc=3)
# 그래프 그리기
plt.show()