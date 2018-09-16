lr = 0.004
arg_lr = 0.004
step_index = 0
epoch_size = 518


def adjust_learning_rate(lr, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (lr - 1e-6) * iteration / (epoch_size * 5)
    else:
        lr = lr * (gamma ** (step_index))

    return lr


lr_all = list()
epoch_all = list()

max_epoch = 300
for iteration in range(0, max_epoch * epoch_size):
    step_values = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)

    if iteration in step_values:
        step_index += 1
    epoch = iteration // epoch_size
    epoch_all.append(epoch)
    lr = adjust_learning_rate(arg_lr, 0.1, iteration // epoch_size, step_index, iteration, epoch_size)
    lr_all.append(lr)

x_all = range(0, max_epoch * epoch_size)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x_all, lr_all, color='red', label='a')
# plt.plot(x_all, epoch_all, color='green')
plt.show()
