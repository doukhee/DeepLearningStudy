#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from IPython.display import display

# 유방암 데이터 가져오기
cancer = load_breast_cancer()
# 학습 데이터 및 테스트 데이터로 분활
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# 결정 트리로 생성
tree = DecisionTreeClassifier(random_state=0)
# 학습 시키기
tree.fit(X_train, y_train)
# 결과 출력
print("Training set score : {:.3f}".format(tree.score(X_train, y_train)))
print("Test set score : {:.3f}".format(tree.score(X_test, y_test)))
# 깊이를 4로만 설정하여 결정 트리 생성
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
# 학습 시키기
tree.fit(X_train, y_train)
# 결과 출력
print("Training set score : {:.3f}".format(tree.score(X_train, y_train)))
print("Test set score : {:.3f}".format(tree.score(X_test, y_test)))
# 그래프 그리기 위한 데이터 생성
export_graphviz(tree, out_file="tree.dot", class_names=["negative", "positive"], feature_names=cancer.feature_names, impurity=False, filled=True)
# 파일 열기
with open("tree.dot") as f:
    # 그래프 데이터 읽어오기
    dot_graph = f.read()
# 그리기 위한 데이터 가져오기
dot = graphviz.Source(dot_graph)
# 포맷을 png로 설정
dot.format = 'png'
# 그리는 파일 이름 설정
dot.render(filename="tree")
# 중요한 속성 출력
print("property import : \r\n{}".format(tree.feature_importances_))

# 그래프로 그리기 위한 함ㅁ쑷
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
    # y 값ㅅ 제한
    plt.ylim(-1, n_features)
    # 그래프 보여주기
    plt.show()
# 그래프 생성
plot_feature_importances_cancer(tree)
# 결정 트리 만들기
tree = mglearn.plots.plot_tree_not_monotone()
# 그래프 보여주기
plt.show()