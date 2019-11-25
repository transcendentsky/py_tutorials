import csv
# import matplotlib.pyplot as plt
import pylab as plt
import numpy as np

# x = plt.linspace(1, 27, 27)
# print(x)

redline = np.array([472,468,465,459,446,419,418,366,260,259,271,259,259,66,379,472,463,453,463,467,469,469,357,294,391,432,447,455], dtype=np.float)
greenline = np.array([472,469,467,460,468,426,426,379,286,286,284,287,287,81,389,472,463,453,463,467,469,469,471,454,455,463,465,456], dtype=np.float)
print(len(redline))
print(len(greenline))
redline = 472.0 - redline
greenline = 472.0 - greenline
print(redline)
redline = redline * 2500000/422
greenline = greenline*2500000/422
# print(redline)
# print(greenline)
x = np.arange(len(redline))
print(x)

fig, ax = plt.subplots()

bar_width = 0.35
# plt.plot(y[:351], vconf1[:351], color='red', label='linear')
plt.bar(x=x-bar_width/2, height=redline, label='blending', alpha=0.4, width=bar_width)
plt.bar(x=x+bar_width/2, height=greenline, color='red', label='non-blending', alpha=0.4, width=bar_width)
# plt.ylim(1.5,4)
plt.grid()
plt.xlabel('layers')
plt.ylabel('num')
plt.legend()
plt.show()
fig.savefig('./comp-w.eps', dpi=600, format='eps')







