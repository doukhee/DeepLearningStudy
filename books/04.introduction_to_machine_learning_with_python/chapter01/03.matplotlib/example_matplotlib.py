#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
# -10에서 10까지 100개의 간격으로 나뉜 배열을 생성
x = np.linspace(-10, 10, 100)
# 사인 함수를 사용하여 y 배열을 생성
y = np.sin(x)
# 플롯 함수는 한 배열의 값을 다른 배열에 대응해서 선 그래프를 그린다
plt.plot(x, y, marker='x')
# 그래프 보여주기
plt.show()