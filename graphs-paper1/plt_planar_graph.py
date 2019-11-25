import pylab as plt
import numpy as np

# def show_scatter(times, epochs, data):
#     # scatter
#     plt.figure(figsize=(8, 5))
#     # 2-dimensions #
#     # plt.scatter(epochs, data, 'o')
#
#     # 3-dimensions #
#     c = np.random.randint(0, 10, 100)
#     plt.scatter(epochs, data, c=c, marker='o')
#     plt.colorbar()
#
#     plt.grid(True)
#     plt.xlabel('epochs')
#     plt.ylabel('data')
#     plt.title('Scatter Plot')
#     plt.show()

def pointgen(x, y):
    sx = 1.1
    sy = 1.7
    offsetx = sx * np.random.rand(16) + x
    offsety = sy * np.random.rand(16) + y
    print(offsetx, offsety)
    return offsetx, offsety


if __name__ == '__main__':
    epochs = np.array(range(100))
    data = np.random.rand(100)  #.reshape((100, 2))


# c1 = [[0.9, 1.1, 1.2, 1.2],[1.4, 1.2,1.0, 1.1]]
# c2 = [[2,3.2,3.3,2], [3.3, 2.2, 2, 3.1]]

fig, ax = plt.subplots(figsize=(8, 5))

x1, y1 = pointgen(2,4.5)
x2, y2 = pointgen(4,1.5)
x3 = 0.75*x1 + 0.25*x2
y3 = 0.75*y1 + 0.25*y2

plt.scatter(x1, y1, c="red", marker='o', label="class 1")
plt.scatter(x2, y2, c="green", marker='>', label="class 2")
plt.scatter(x3[:6], y3[:6], c="blue", marker='o', label="blended class")
# plt.ylim(0.5,3.5)
# plt.xlim(0.5,3.5)
plt.legend()
plt.xticks([])
plt.yticks([])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
fig.savefig('./scalars.eps', dpi=600, format='eps')