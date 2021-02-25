#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import mglearn 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
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
# 학습 시킬 데이터 분활
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# 랜덤 포레스트로 분류
# n_estimators 는 생성할 트리 갯수를 설정하는 옵션
forest = RandomForestClassifier(n_estimators=100, random_state=0)
# 학습 하기
forest.fit(X_train, y_train)
# 정확도 출력
print("Accuracy on training set : {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set : {:.3f}".format(forest.score(X_test, y_test)))

# 그래프 그리기
plot_feature_importances_cancer(forest)
