import os
import numpy as np
from itertools import groupby
from skimage import morphology,measure
from PIL import Image
from scipy import misc

# 因为一张图片里只有一种类别的目标，所以label图标记只有黑白两色
rgbmask = np.array([[0,0,0],[255,255,255]],dtype=np.uint8)

# 从label图得到 boundingbox 和图上连通域数量 object_num
def getboundingbox(image):
    # mask.shape = [image.shape[0], image.shape[1], classnum]
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[np.where(np.all(image == rgbmask[1],axis=-1))[:2]] = 1
    # 删掉小于10像素的目标
    mask_without_small = morphology.remove_small_objects(mask,min_size=10,connectivity=2)
    # 连通域标记
    label_image = measure.label(mask_without_small)
    #统计object个数
    object_num = len(measure.regionprops(label_image))
    boundingbox = list()
    for region in measure.regionprops(label_image):  # 循环得到每一个连通域bbox
        boundingbox.append(region.bbox)
    return object_num, boundingbox