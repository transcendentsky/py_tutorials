# coding:utf-8

"""
def plt_bboxes(img, rclasses, rscores, rbboxes, \
               nms_rclasses, nms_rscores, nms_rbboxes, \
               all_rclasses, all_rscores, all_rbboxes, \
               gbboxes, figsize=(10,10), linewidth=1.5):
    Visualize bounding boxes. Largely inspired by SSD-MXNET!

    print('plt.show')
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()

    def plt_r():
        '''
        '''
        print('rclass.shape: ')
        print(rclasses.shape)
        print("rclass id: ")
        print(rclasses[0])
        for i in range(rclasses.shape[0]):
            cls_id = int(rclasses[i])
            if cls_id >= 0:
                score = rscores[i]
                if cls_id not in colors:
                    colors[cls_id] = (random.random(), random.random(), random.random())
                ymin = int(rbboxes[i, 0] * height)
                xmin = int(rbboxes[i, 1] * width)
                ymax = int(rbboxes[i, 2] * height)
                xmax = int(rbboxes[i, 3] * width)
                rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                     ymax - ymin, fill=False,
                                     #edgecolor=colors[cls_id],
                                     edgecolor=(0.9,0.9,0.3),
                                     linewidth=linewidth)
                plt.gca().add_patch(rect)
"""

import matplotlib.pyplot as plt
import numpy as np


def plt_tables(times, epochs, data):
    print("plt show")
    fig = plt.figure(figsize=None)


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


def show_scatter(times, epochs, data):
    # scatter
    plt.figure(figsize=(8, 5))
    # 2-dimensions #
    # plt.scatter(epochs, data, 'o')

    # 3-dimensions #
    c = np.random.randint(0, 10, 100)
    plt.scatter(epochs, data, c=c, marker='o')
    plt.colorbar()

    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('data')
    plt.title('Scatter Plot')
    plt.show()


def show_hist(times, epochs, data):
    # histogram
    plt.figure(figsize=(8, 5))
    # ValueError: `bins` must increase monotonically, when an array
    # plt.hist(data, epochs,label=['1st', '2nd'])
    plt.hist(data, 100, label=['1st'])
    plt.grid(True)
    plt.legend(loc=0)
    # 表示最佳位置显示图例
    plt.xlabel('epochs')
    plt.ylabel('data')
    plt.title('Test')
    plt.show()

    # 两个数据集堆叠的直方图。
    # plt.hist(y, bins=20, label=['1st', '2nd'], color=['b', 'm'], stacked=True, rwidth=0.8)
    # 参数stacked = True表示堆叠的直方图


def others():
    '''
    plt.boxplot(y)     表示用y绘制箱形图；
    plt.grid(True)      表示添加网格；
    plt.setp(ax, xticklabels=['1st', '2nd']) 表示刻度值标签设置为'1st' 和 '2nd'
    '''

    pass


if __name__ == '__main__':
    epochs = np.array(range(100))
    # epochs = np.random.rand(100)
    # data = np.random.rand(100)
    data = np.random.rand(200).reshape((100, 2))

    show_plot(None, epochs, data)
    # show_scatter(None, epochs, data)
    # show_hist(None, epochs, data)
