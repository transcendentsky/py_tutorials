import numpy as np

x = np.arange(0,50000)

np.random.shuffle(x)
limited_x = x[:25000].copy()
np.random.shuffle(x)
limited_x2 = x[:25000].copy()



v = np.arange(0,25000)
print(limited_x)


# random index
ids = np.random.randint(0,25000,32)
print(id)

index = np.array([2,2,2,22,2])
index2 = [3,2,23,3,4]

print(len(index2))
print(index.size)

print(np.random.random())


xx = limited_x[ids]
print(xx)