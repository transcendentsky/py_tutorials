# coding: utf-8

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

## Dataset pipeline
#  Resample to the same spacing
#  cut off to [-79, 304] , and Z-Score
#  remove some bad instances and correct some wrong labels

class LswLoader(Dataset):
    """
    LSW dataset:
    Image Type: DICOM and nrrd
    """
    def __init__(self, load_mod="all", img_dir="/home1/quanquan/datasets/lsw/benign_65/fpAML_55/"):
        ""
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
        
        if self.load_mod == "sitk_image":
            return image, nrrd_image
        
        raise TypeError("No valid load_mod, but GOT: {}".format(load_mod))
        
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
                    
        
def resampleImage(Image:sitk.SimpleITK.Image, SpacingScale=None, NewSpacing=None, NewSize=None, Interpolator=sitk.sitkLinear)->sitk.SimpleITK.Image:
    """
    Author: Pengbo Liu
    Function: resample image to the same spacing
    Params:
        Image, SITK Image
        SpacingScale / NewSpacing / NewSize , are mutual exclusive, independent.
    """
    Size = Image.GetSize()
    Spacing = Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()

    if not SpacingScale is None and NewSpacing is None and NewSize is None:
        NewSize = [int(Size[0]/SpacingScale),
                   int(Size[1]/SpacingScale),
                   int(Size[2]/SpacingScale)]
        NewSpacing = [Spacing[0]*SpacingScale,
                      Spacing[1]*SpacingScale,
                      Spacing[2]*SpacingScale]
        print('Spacing old: [{:.3f}, {:.3f}, {:.3f}] Spacing new: [{:.3f}, {:.3f}, {:.3f}]'.format(Spacing[0], Spacing[1], Spacing[2], NewSpacing[0], NewSpacing[1],  NewSpacing[2]))
    elif not NewSpacing is None and SpacingScale is None and NewSize is None:
        NewSize = [int(Size[0] * Spacing[0] / NewSpacing[0]),
                   int(Size[1] * Spacing[1] / NewSpacing[1]),
                   int(Size[2] * Spacing[2] / NewSpacing[2])]
        print('Spacing old: [{:.3f}, {:.3f}, {:.3f}] Spacing new: [{:.3f}, {:.3f}, {:.3f}]'.format(Spacing[0], Spacing[1], Spacing[2], NewSpacing[0], NewSpacing[1], NewSpacing[2]))
    elif not NewSize is None and SpacingScale is None and NewSpacing is None:
        NewSpacing = [Spacing[0]*Size[0] / NewSize[0],
                      Spacing[1]*Size[1] / NewSize[1],
                      Spacing[2]*Size[2] / NewSize[2]]
        print('Spacing old: [{:.3f}, {:.3f}, {:.3f}] Spacing new: [{:.3f}, {:.3f}, {:.3f}]'.format(Spacing[0],Spacing[1],Spacing[2],NewSpacing[0],NewSpacing[1],NewSpacing[2]))


    Resample = sitk.ResampleImageFilter()
    Resample.SetOutputDirection(Direction)
    Resample.SetOutputOrigin(Origin)
    Resample.SetSize(NewSize)
    Resample.SetOutputSpacing(NewSpacing)
    Resample.SetInterpolator(Interpolator)
    NewImage = Resample.Execute(Image)

    return NewImage        


if __name__ == "__main__":
    dataloader = LswLoader(load_mod="sitk_image")
    # print(dataloader.image_name)
    # input()
    # print(dataloader.label_name)
    # data2slices("/home1/quanquan/datasets/lsw/benign_65/fpAML_55/")
    
    image, nrrd_image = dataloader.__getitem__(0)
    assert type(image) == sitk.SimpleITK.Image
    print(type(image))