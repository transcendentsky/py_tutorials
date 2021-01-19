import torch
import numpy as np
import os
import monai
from monai.data import ArrayDataset, create_test_image_2d
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, AddChannel, AsDiscrete, Compose, LoadImage, RandRotate90, RandSpatialCrop, ScaleIntensity, ToTensor, LoadNumpy, LoadNifti
from monai.visualize import plot_2d_or_3d_image

from torch.utils.data import Dataset, DataLoader
from tutils import *

tconfig.set_print_info(True)

train_imtrans = Compose(
    [
        ToTensor(), 
        AddChannel(),
        RandSpatialCrop((96, 96), random_size=False),
    ]
)
#  RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        # AddChannel(),
        # ToTensor(),
        # ScaleIntensity(),
        # AddChannel(),
        # RandSpatialCrop((96, 96), random_size=False),
# LoadNifti(),
train_segtrans = Compose(
    [
        LoadNifti(),
        AddChannel(),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ToTensor(),
    ]
)
# For testing 
# datadir1 = "/home1/quanquan/datasets/lsw/benign_65/fpAML_55/slices/"
# image_files = np.array([x.path for x in os.scandir(datadir1+"image") if x.name.endswith(".npy")])
# label_files = np.array([x.path for x in os.scandir(datadir1+"label") if x.name.endswith(".npy")])

###  Data Collection for Kits19 
datadir_kits = "/home1/quanquan/datasets/kits19/resampled_data"
image_files = []
for subdir in os.scandir(datadir_kits):
    if subdir.name.startswith("case_"):
        image_name = os.path.join(subdir.path, "imaging.nii.gz")
        image_files.append(image_name)
image_files = np.array(image_files)
image_files.sort()
print(image_files[:10])
   
### Define array dataset, data loader
# check_ds = ArrayDataset(img=image_files, img_transform=train_imtrans, seg=None, seg_transform=None)
check_ds = monai.data.NiftiDataset(image_files=image_files, transform=train_imtrans)
check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
im = monai.utils.misc.first(check_loader)
print(im.shape)

### Create a training data loader
train_ds = ArrayDataset(image_files[:-20], train_imtrans, seg=None, seg_transform=None, label=None, label_transform=None)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())

p("Start Training")
for idx, batch_data in enumerate(train_loader):
    p("len(batch_data): ", len(batch_data))
    # inputs, labels = batch_data[0].cuda(), batch_data[1].cuda()
    inputs = batch_data
    import ipdb; ipdb.set_trace()
    break
    
    