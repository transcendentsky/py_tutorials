import numpy as np
import matplotlib.pyplot as plt

# theta = np.array([0.25, 0.75,1,1.5, 0.25])
labels = np.arange(32)
theta = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
# r = [20,60,40,80,20]
values = np.random.randint(16,33,size=32)

theta = np.concatenate((theta, [theta[0]]))
values = np.concatenate((values, [values[0]]))

# plt.polar(theta*np.pi, r ,'ro-', lw=2)
# plt.ylim(0,100)
# plt.show()

ax = plt.subplot(111, projection='polar')
ax.plot(theta, values, 'm-', lw=2, alpha=0.5)
ax.set_thetagrids(theta*180/np.pi, labels)
ax.set_ylim(0,40)
plt.show()