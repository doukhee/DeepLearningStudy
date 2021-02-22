#/!usr/bin/env python3
#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 가상 데이터 만들기
X, y = make_blobs(random_state=42)
# 선점도 만들기
mglearn.discrete_scatter(X[:,0], X[:, 1], y)
# x와 y라벨 생성
plt.xlabel("property 0")
plt.ylabel("property 1")
# 범주 등록
plt.legend()
# 그래프 보이기
plt.show()