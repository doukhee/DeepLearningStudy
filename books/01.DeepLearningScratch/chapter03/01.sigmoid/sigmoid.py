# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.array([-1.0, 1.0, 2.0])

y = sigmoid(x)

print("sigmoid : {}".format(y))

t = np.array([1.0, 2.0, 3.0])
print("Array Scar {}".format(1.0 + t))
print("Array Scar {}".format(1/t))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()