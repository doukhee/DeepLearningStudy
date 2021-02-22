#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# 데이터 초기화
X, y = mglearn.datasets.make_forge()
# 화면 분활 및 화면 크기 추가
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
#
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    #
    clf = model.fit(X, y)
    #
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
    #
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    # 타이틀 추가
    ax.set_title("{}".format(clf.__class__.__name__))
    # x 라벨 추가
    ax.set_xlabel("property 0")
    # y 라벨 추가
    ax.set_ylabel("property 1")
# 범주 추가
axes[0].legend()
# 차트 그리기
plt.show()
# svc 분류로 사용
mglearn.plots.plot_linear_svc_regularization()
# 차트 보여주기
plt.show()