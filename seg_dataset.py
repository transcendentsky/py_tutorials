import os
import numpy as np
import torch
import random
import torchvision
import string

import random
import time
import cv2
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk

from tutils import *

class LswLoader(Dataset):
    """
    LSW dataset:
    Image Type: DICOM and nrrd
    """
    def __init__(self, load_mod="all", img_dir="/home1/quanquan/datasets/lsw/benign_65/fpAML_55/"):
        #self.super(LswDataset, self).__init__()
        self.img_dir  = img_dir
        self.load_mod = load_mod
        
        self.image_name = np.array([x.name for x in os.scandir(img_dir) if (os.path.isdir(x.path) and x.name != "slices")])
        self.label_name = np.array([x.name for x in os.scandir(img_dir) if x.name.endswith(".nrrd")])
        
        self.image_name.sort()
        self.label_name.sort()
        
    def __getitem__(self, index):
        image_name = self.image_name[index]
        label_name = self.label_name[index]
        # ----------------  DICOM Image  -----------------
        dicom_path = os.path.join(self.img_dir, image_name)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        image_np = sitk.GetArrayFromImage(image)      # (c, h, w)
        # ----------------  NRRD LABEL  -------------------
        nrrd_path  = os.path.join(self.img_dir, label_name)
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NrrdImageIO")
        reader.SetFileName(nrrd_path)
        nrrd_image = reader.Execute()
        # image_array = sitk.GetArrayViewFromImage(nrrd_image)
        label_np = sitk.GetArrayFromImage(nrrd_image)  # (c, h, w)
        
        if self.load_mod == "all":
            return torch.from_numpy(image_np).float(), \
                torch.from_numpy(label_np).float()
        
        if self.load_mod == "slice":
            return torch.from_numpy(image_np[0,:,:][np.newaxis,:,:]).float(), \
                torch.from_numpy(label_np[0,:,:][np.newaxis,:,:]).float()
                
        if self.load_mod == "original":
            return image_np, label_np
        
    def __len__(self):
        assert len(self.image_name) == len(self.label_name)
        return len(self.image_name)
        
        
def data2slices(img_dir):
    dataset = LswLoader(load_mod="original")
    # if texists(img_dir, "slices"):
    #     print("Dir exists")
    #     return 
    for i in range(len(dataset)):
        image_np, label_np = dataset.__getitem__(i)
        channel = image_np.shape[0]
        print("\r Writing images and labels \t[{}/{}]\t ....".format(i, len(dataset)), end="")
        for j in range(channel):
            # seg_pixel = np.sum(label_np[j,:,:])
            # if seg_pixel <= 100:
            #     continue
            # print("np.sum: ", seg_pixel)
            if not os.path.exists(tfilename(img_dir, "slices/image", "image_{}_{}.png".format(i, j))):
                if not os.path.exists(tfilename(img_dir, "slices/label", "label_{}_{}.png".format(i, j))):
                    seg_pixel = np.sum(label_np[j,:,:])
                    if seg_pixel <= 100:
                        continue
                    print("np.sum: ", seg_pixel)
                    # cv2.imwrite(tfilename(img_dir, "slices/image", "image_{}_{}.png".format(i, j)), image_np[j,:,:])
                    # cv2.imwrite(tfilename(img_dir, "slices/label", "label_{}_{}.png".format(i, j)), label_np[j,:,:])
                    np.save(tfilename(img_dir, "slices/image", "image_{}_{}.npy".format(i,j)), image_np[j,:,:])
                    np.save(tfilename(img_dir, "slices/label", "image_{}_{}.npy".format(i,j)), label_np[j,:,:])
                    
        
if __name__ == "__main__":
    # dataloader = LswLoader()
    # print(dataloader.image_name)
    # input()
    # print(dataloader.label_name)
    data2slices("/home1/quanquan/datasets/lsw/benign_65/fpAML_55/")