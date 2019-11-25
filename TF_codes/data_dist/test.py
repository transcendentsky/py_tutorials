
# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.random.beta(0.1,0.1)
# y = []
# for i in range(1000):
#     y.append(np.random.beta(0.1,0.1))
# y = np.array(y)
# x = np.ones_like(y)
#
# plt.figure()
# plt.scatter(y,x)
# plt.show()

from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

a, b = 0.01, 0.01

# mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')

x = np.linspace(0, 1, 100)
plt.plot(x, beta.pdf(x, 0.1, 0.1), 'r-', lw=1, alpha=0.6, label='beta 0.01', color='blue')
plt.plot(x, beta.pdf(x, 0.1, 10), 'r-', lw=1, alpha=0.6, label='beta 0.1', color='red')
plt.plot(x, beta.pdf(x, 0.1, 20), 'r-', lw=1, alpha=0.6, label='beta 0.1', color='green')

for i in range(10):
    v = -i*0.01
    # plt.plot(x, beta.pdf(x, 0.1+v, 0.1+v), 'r-', lw=1, alpha=0.6, label='beta {}'.format(0.1+v), color='red')

plt.legend()
plt.show()

import numpy as np

# for i in range(10):
#     print(np.random.randint(0,5,(2,)))
_locs = [[3,4],
        [2, 2],
        [3, 1],
        [1, 0],
        [3, 1],
        [3, 1],
        [4, 0],
        [4, 1],
        [0, 2],
        [4, 3]]

print(_locs)
print(np.random.randint(0,10))