#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
# 넘파이의 배열 생성
X = np.array([[0, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 1, 0]])
# 넘파이의 배열 생성
y = np.array([0, 1, 0, 1])
# 카운트 값을 담기 위한 dictionary 변수
counts = {}
# unique(x) : 배열 내 중복된 원소 제거 후 유일한 원소를 정렬하여 반환
# 반복을 하여 dictionary 에 값을 저장하기 위한 반복문
for label in np.unique(y):
    counts[label] = X[y == label].sum(axis=0)
#  특성을 카운트한 값을 출력
print("Property counter :\r\n {}".format(counts))