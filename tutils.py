from collections import OrderedDict
import os
import numpy as np
import torch
import random
import torchvision
import string

import random
import time
import cv2

from pathlib import Path

TUTILS_DEBUG = True

def tt():
    # print("[Trans Utils] ", end="")
    pass

def p(*s,end="\n", **kargs ):
    print("[Trans Utils] ", s, kargs, end="")
    print("", end=end)

def d(*s,end="\n", **kargs ):
    if TUTILS_DEBUG:
        print("[Trans Utils] ", s, kargs, end="")
        print("", end=end)

def time_now():
    tt()
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def generate_random_str(n):
    tt()
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, n))
    return ran_str

def generate_name():
    tt()
    return time_now() + generate_random_str(6)
    
def write_image_np(image, filename):
    tt()
    cv2.imwrite("wc_" + generate_random_str(5)+'-'+ time_now() +".jpg", image.astype(np.uint8))
    pass

def tdir(dir_path):
    tt()
    if not os.path.exists(dir_path):
        print("Create Dir Path: ", dir_path)
        os.makedirs(dir_path)

    return dir_path

def tfilename(*filenames):
    filename = os.path.join(*filenames)
    d(filename)
    parent, name = os.path.split(filename)
    if not os.path.exists(parent):
        os.makedirs(parent)
    return filename


if __name__ == "__main__":
    tfilename("dasd", "dasdsa")

