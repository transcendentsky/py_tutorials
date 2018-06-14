import matplotlib.pyplot as plt

def plt_bboxes(img, rclasses, rscores, rbboxes, \
               nms_rclasses, nms_rscores, nms_rbboxes, \
               all_rclasses, all_rscores, all_rbboxes, \
               gbboxes, figsize=(10, 10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
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
                                     # edgecolor=colors[cls_id],
                                     edgecolor=(0.9, 0.9, 0.3),
                                     linewidth=linewidth)
                plt.gca().add_patch(rect)

    #                class_name = str(cls_id)
    #                plt.gca().text(xmin, ymin - 2,
    #                               '{:s} | {:.3f}'.format(class_name, score),
    #                               bbox=dict(facecolor=colors[cls_id], alpha=0.5),
    #                               fontsize=12, color='white')


    def plt_gb():
        for i in range(len(gbboxes)):
            #           print(gbboxes)
            xmin = gbboxes[i][0]
            ymin = gbboxes[i][1]
            xmax = gbboxes[i][2]
            ymax = gbboxes[i][3]
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=(1, 1, 1),
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)

    plt_r()
    #    plt_nmsr()
    #    plt_allr()
    #    plt_gb()

    plt.show()
    fig.savefig('test.pdf')