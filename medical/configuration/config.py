import configparser # python 3.x
# import ConfigParser # python 2.x
from tutils import *

# config = configparser.ConfigParser()

# # 用法1
# config["DEFAULT"] = {
#     "Country": "China",
#     "Max_Iter": "2",
# }

# # 用法2
# config["section_1"] = {}
# config["section_1"]["a"] = "2"

# # 用法3
# config["section_2"] = {}
# section2 = config["section_2"]
# section2["b"] = "2"

# config['DEFAULT']['ForwardX11'] = 'yes'

# with open("example.ini", "w") as fp:
#     config.write(fp)
    

import yaml



from tensorboardX import SummaryWriter

output_dir = tdir("output_test/wc/" , generate_name() + "train/")  # train
output_dir = tdir("output_test/wc/" , generate_name() + "test/")   # test

writer = SummaryWriter(logdir=tdir(output_dir, "summary"))

fname = tfilename(output_dir, "img/", epoch, "wc_ori_pred"+".jpg") # pred     
fname = tfilename(output_dir, "img/", epoch, "wc_ori_gt"+".jpg")   # Gt

torch.save(state, tfilename(output_dir+"model/{}_{}_{}_{}_model.pkl"\
    .format(args.arch, epoch+1,train_mse,experiment_name)))


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.RETINANET_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.FBNET = CN()
_C.MODEL.FBNET.ARCH = "default"
# custom arch
_C.MODEL.FBNET.ARCH_DEF = ""
_C.MODEL.FBNET.BN_TYPE = "bn"