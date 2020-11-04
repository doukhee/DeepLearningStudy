# coding: utf-8

a = [1,2,3,4,5]
print(a)

print(len(a))

print(a[0])

print(a[4])

a[4] = 99

print("{} list ".format(a))

print("array is a[0~2]: {}".format(a[0:2]))

print("array is a[1~4] : {}".format(a[1:]))

print("array is a[0~3] : {}".format(a[:3]))

print("array is a[0~ last - 1] : {}".format(a[:-1]))

print("array is a[0 ~ last - 2] : {}".format(a[:-2]))
