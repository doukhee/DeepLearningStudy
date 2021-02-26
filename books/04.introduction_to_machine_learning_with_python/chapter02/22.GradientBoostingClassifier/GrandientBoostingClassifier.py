#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# 그래프로 그리기 위한 함수
def plot_feature_importances_cancer(model):
    # 데이터의 수 가져오기
    n_features = cancer.data.shape[1]
    # 가로 막대 그래프 생성
    plt.barh(range(n_features), model.feature_importances_, align='center')
    # y축 눈금 값 설정
    plt.yticks(np.arange(n_features), cancer.feature_names)
    # 라벨 설정
    plt.xlabel("property importance")
    plt.ylabel("property")
    # y 값 제한
    plt.ylim(-1, n_features)
    # 그래프 보여주기
    plt.show()

# 유방암 데이터 가져오기
cancer = load_breast_cancer()
# 테스트 데이터와 학습 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# 그래디언트 부스팅 회귀 트리로 분류 하기
gbrt = GradientBoostingClassifier(random_state=0)
# 학습 하기
gbrt.fit(X_train, y_train)
# 학습률 및 테스트 스코어 출력
print("Train set accuracy : {:.3f}".format(gbrt.score(X_train, y_train)))
print("Test set accuracy : {:.3f}\r\n".format(gbrt.score(X_test, y_test)))
# 그래디언트 부스팅 회귀 트리로 분류 하기 최대 깊이는 1로 설정
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
# 학습 하기
gbrt.fit(X_train, y_train)
# 학습률 및 테스트 스코어 출력
print("Max depth 1 Train set accuracy : {:.3f}".format(gbrt.score(X_train, y_train)))
print("Max depth 1 Test set accuracy : {:.3f}\r\n".format(gbrt.score(X_test, y_test)))
# 확인을 위한 그래프 출력
plot_feature_importances_cancer(gbrt)

# 그래디언트 부스팅 회귀 트리로 분류 하기 학습률을 0.01로 설정
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
# 학습하기
gbrt.fit(X_train, y_train)
# 학습률 및 테스트 스코어 출력
print("Learing Rate 0.01 Train set accuracy : {:.3f}".format(gbrt.score(X_train, y_train)))
print("Learing Rate 0.01 Test set accuracy : {:.3f}\r\n".format(gbrt.score(X_test, y_test)))

