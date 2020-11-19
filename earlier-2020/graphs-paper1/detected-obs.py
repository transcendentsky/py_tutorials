import csv
# import matplotlib.pyplot as plt
import pylab as plt
import numpy as np

x = plt.linspace(0, 0.95, 20)
print(x)
redline = np.array([341, 168, 104, 79, 66, 55, 47, 43, 38, 29, 17, 10, 6, 6,5,5,7,8,8,10], dtype=np.float)
greenline = np.array([279,146,96,72,59,47,40,35,32,25,16,10,7,6,6,7,7,8,9,8], dtype=np.float)
print(len(redline))
print(len(greenline))
redline = redline * 8000/322
greenline = greenline*8000/322



fig, ax = plt.subplots()

# plt.plot(y[:351], vconf1[:351], color='red', label='linear')
plt.plot(x, redline, color='red', label='blending')
plt.plot(x, greenline, color='green', label='non-blending')
# plt.plot(y2, vconf4, color="green", label="exp")
# plt.ylim(1.5,4)
plt.grid()
plt.xlabel('thresholds')
plt.ylabel('num')
plt.legend()
plt.show()
fig.savefig('./detect-obs.eps', dpi=600, format='eps') # save eps figure