#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression().fit(X_train, y_train)

print("Train set score : {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score : {:.2f}".format(lr.score(X_test, y_test)))

# 구분을 위한 출력
print("")

ridge = Ridge().fit(X_train, y_train)
print("Train set score : {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score : {:.2f}".format(ridge.score(X_test, y_test)))