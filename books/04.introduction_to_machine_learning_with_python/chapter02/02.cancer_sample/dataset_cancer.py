#!/usr/bin/env python3
#-*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
# 유방암 데이터를 가져오기
cancer = load_breast_cancer()
# 유방암에 있는 목록들 출력
print("cancer.keys() : \r\n{}".format(cancer.keys()))
# 구분을 위한 한줄 띄기
print("")
# 유방암 데이터의 모양 출력
print("Cancer Breast shape : {}".format(cancer.data.shape))
# 구분을 위한 한줄 띄기
print("")
# 클래스별 샘플 데이터 출력
print("Class Sample Number : \r\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
# 구분을 위한 한줄 띄기
print("")
# 유방암 데이터의 특성을 출력
print("property name : \r{}".format(cancer.feature_names))
# 구분을 위한 한줄 띄기
print("")
