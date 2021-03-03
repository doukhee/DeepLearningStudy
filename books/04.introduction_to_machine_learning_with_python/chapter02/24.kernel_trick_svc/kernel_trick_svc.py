#!/usr/bin/env python3
#-*- coding: utf-8 -*-
from sklearn.svm import SVC
import mglearn
import matplotlib.pyplot as plt

X, y = mglearn.tools.make_handcrafted_dataset()
# SVM으로 학습 시키기
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
# 2차원 선점도 그리기
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
# 선점도 그리기
mglearn.discrete_scatter(X[:,0], X[:, 1], y)
# 서포트 벡터
sv = svm.support_vectors_
# dual_coef_의 부호에 의흐ㅐ 서포트 벡터의 클래스 레이블이 결정
sv_labels = svm.dual_coef_.ravel() > 0
# 선점도 그리기
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
# 라벨 설정
plt.xlabel("property 0")
plt.ylabel("property 1")
# 그래프 그리기
plt.show()
# 화면 분활 및 화면 생성
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
#  C값을 증가 시키기 위한 반복문
for ax, C in zip(axes, [-1, 0, 3]):
    # 감마 값을 변경하기 위한 반복문
    for a, gamma in zip(ax, range(-1, 2)):
        # 감마 값과 C 값을 변경하여 그래프 만들기
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
# 범주 생성
axes[0, 0].legend(["class 0", 'class 1', 'class 0 support vector', 'class 1 support vector'], ncol=4, loc=(.9, 1.2))
# 그래프 보이기
plt.show()
