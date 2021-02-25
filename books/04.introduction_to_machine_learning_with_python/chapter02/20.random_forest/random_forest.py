#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import mglearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
# 데이터 생성 샘플링 수는 100개, 잡음은 0.25로 설정
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# 학습데이터와 테스트 데이터 분류 stratify는 class의 비율을 설정하는 flag
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# 랜덤 포레스트 분류 모델 만들기
forest = RandomForestClassifier(n_estimators=5, random_state=2)
# 학습하기
forest.fit(X_train, y_train)
# 화면 분활 하기
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
# ravel()함수는 일차원 함수로 펴주는 함수이다
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    # 그래프 타이틀 설정
    ax.set_title("tree {}".format(i))
    # 그래프 그리기
    mglearn.plots.plot_tree_partition(X, y, tree, ax=ax)
# 2차원 선점도 그리기?
mglearn.plots.plot_2d_separator(forest, X, fill=True, ax=axes[-1, -1], alpha=.4)
# 차트 타이틀 설정
axes[-1, -1].set_title("Randorm forest")
# 결정 선점도 그리기
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# 그래프 보여주기
plt.show()