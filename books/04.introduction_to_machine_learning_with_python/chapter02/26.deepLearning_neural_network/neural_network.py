#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import mglearn
# -3부터 3까지 100개의 값을 임의로 가져오기
line = np.linspace(-3, 3, 100)
# 그래프 생성
plt.plot(line, np.tanh(line), label='tanh')
plt.plot(line, np.maximum(line, 0), label='relu', linestyle='--')
# 범주 생성
plt.legend(loc='best')
# 라벨 설정
plt.xlabel('x')
plt.ylabel('relu(x), tanh(x)')
# 그래프 보여주기
plt.show()

mglearn.plots.plot_two_hidden_layer_graph()

