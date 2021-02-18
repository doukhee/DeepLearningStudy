#!/usr/bin/env python3
#-*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
# 데이터 셋을 생성
X, y = mglearn.datasets.make_forge()
# 산점도를 그린다
mglearn.discrete_scatter(X[:,0], X[:,1], y)
# 레전드 설정
plt.legend(['class 0', 'class 1'], loc=4)
# x 라벨 설정
plt.xlabel("First Property")
# y 라벨 설정
plt.ylabel('Second Property')
# x데이터의 모양 출력
print("X.shape : {}".format(X.shape))
# 차트 보여주기
plt.show()