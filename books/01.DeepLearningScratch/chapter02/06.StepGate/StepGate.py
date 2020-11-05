# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

def Step_Function1(x):
    if x > 0:
        return 1
    else:
        return 0

def STEP_Function(x):
    y = x > 0
    # 형변환하는 numpy 함수
    return y.astype(np.int)


x = np.arange(-5.0, 5.0, 0.1)

y = STEP_Function(x)

plt.plot(x, y)

plt.ylim(-0.1, 1.1)
plt.show()