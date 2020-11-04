# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("../../asserts/lena.png")

plt.imshow(img)

plt.show()
