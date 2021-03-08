#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
# 데이터 셋 가져오기
x, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# 데이터 분류 해서 테스트 데이터 및 훈련 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)
# 다층퍼셉트론 분류기로 학습 시키기
mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
# 2차원 선점도 그리기
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# 선점도 그리기
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# 라벨 설정
plt.xlabel("property 0")
plt.ylabel("property 1")
# 그래프 보여주기
plt.show()

# 다층퍼셉트론으로 학습 진행 10개의 유닛으로 설정된 은닉층
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10]).fit(X_train, y_train)

# 2차원 선점도 그리기
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# 선점도 그리기
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# 라벨 설정
plt.xlabel("property 0")
plt.ylabel("property 1")
# 그래프 그리기
plt.show()

# 다층퍼셉트론으로 학습 진행 10개의 유닛으로 설정된 은닉층 2로 학습 진행
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10]).fit(X_train, y_train)

# 2차원 선점도 그리기
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# 선점도 그리기
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# 라벨 설정
plt.xlabel("property 0")
plt.ylabel("property 1")
# 그래프 그리기
plt.show()

# 다층퍼셉트론으로 학습 진행 10개의 유닛으로 설정된 은닉층 2로 학습 진행 활성 함수를 탄젠트로 설정
mlp = MLPClassifier(solver='lbfgs', activation='tanh', random_state=0, hidden_layer_sizes=[10, 10]).fit(X_train, y_train)

# 2차원 선점도 그리기
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# 선점도 그리기
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# 라벨 설정
plt.xlabel("property 0")
plt.ylabel("property 1")
# 그래프 그리기
plt.show()

# 화면 생성 및 분활
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
#
for axx, n_hidden_nodes in zip(axes, [10, 100]):
    #
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
        # 다층퍼셉트론으로 학습 진행 10개의 유닛으로 설정된 은닉층 2로 학습 진행
        mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha, max_iter=1000).fit(X_train, y_train)
        # 2차원 선점도 그리기 분활된 화면을 설정 해주는 ax
        mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
        # 선점도 그리기 분활된 화면을 설정 해주는 ax
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
        # 그래프 타이틀 설정
        ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha))
# 그래프 그리기
plt.show()
