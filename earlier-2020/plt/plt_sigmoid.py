

import matplotlib.pyplot as plt
# import matplotlib.numerix as nx
import numpy as np

def show_plot(times, epochs, data):
    # 折线图 Or Scatter
    plt.figure(figsize=(8, 5))
    """
    args:
    marker='o' ,'x',
    color=
    """

    plt.plot(epochs, data[:, 0], color='red', label='0')
    plt.plot(epochs, data[:, 1], color='green', marker='x', label='1')
    plt.legend()  # 显示图例
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('data')
    plt.title('Test')
    plt.show()


# import section
from matplotlib import pylab
import pylab as plt
import numpy as np
###
### Pylab : another interface of matplotlib , similiar as pyplot

# sigmoid = lambda x: 1 / (1 + np.exp(-x))
def sigmoid(x):
    n = 200
    alpha = .2
    return (1 / (1 + np.exp(-(x-n)*alpha)))


mySamples = []
mySigmoid = []

fig, ax = plt.subplots()

# generate an Array with value ???
# linespace generate an array from start and stop value
# with requested number of elements. Example 10 elements or 100 elements.
#
# x = plt.linspace(0, 400, 300)
y = plt.linspace(0, 400, 300)
print(y)

# prepare the plot, associate(ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24) the color r(ed) or b(lue) and the label
# plt.plot(x, sigmoid(x), 'r', label='linspace(-10,10,10)')
plt.plot(y, sigmoid(y), 'b', label='linspace(-10,10,100)')

# Draw the grid line in background.
plt.grid()

# Title & Subtitle
# plt.title('Sigmoid Function')
# plt.suptitle('Sigmoid')

# place the legen boc in bottom right of the graph
# plt.legend(loc='lower right')

# write the Sigmoid formula
# plt.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=15)
plt.text(4, 0.85, r'$\lambda = \frac{\hat{\lambda}}{1+e^{-\alpha(epoch-n)}}$', fontsize=18)
# resize the X and Y axes
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(50))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))

# plt.plot(x)
plt.ylabel('$\lambda$')
plt.xlabel('epoch')

# create the graph
plt.show()
fig.savefig('./sigmoid.eps', dpi=600, format='eps')