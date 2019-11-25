import csv
# import matplotlib.pyplot as plt
import pylab as plt
import numpy as np

# x = plt.linspace(1, 27, 27)
# print(x)

redline = np.array([454,460,434,451,424,417,422,377,373,412,386,382,371,206,77,468,470,469,468,468,468,430,426,426,426,441,446], dtype=np.float)
greenline = np.array([465,460,461,444,421,428,423,370,373,334,360,364,375,342,471,467,465,466,467,470,470,465,470,470,470,470,465], dtype=np.float)
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
# plt.xticks(x + 0.35 / 2)
plt.xlabel('layers')
plt.ylabel('num')
plt.legend()
plt.show()
fig.savefig('./comp-b.eps', dpi=600, format='eps')





