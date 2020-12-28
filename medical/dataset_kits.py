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
from tqdm import tqdm


from tutils import *

## Dataset pipeline
#  Resample to the same spacing
#  cut off to [-79, 304] , and Z-Score
#  remove some bad instances and correct some wrong labels

class Kits19(Dataset):
    
    def __init__(self, load_mod:str="all", datadir:str="/home1/quanquan/datasets/kits19/data"):
        self.datadir  = datadir
        self.load_mod = load_mod
        
        self.dirnames = np.array([x.name for x in os.scandir(datadir) if (os.path.isdir(x.path) and x.name.startswith("case_"))])
        self.dirnames.sort()
        
    def __getitem__(self, index):
        image_name = "imaging.nii.gz"
        label_name = "segmentation.nii.gz"
        
        image_path = os.path.join(self.datadir, self.dirnames[index], image_name)
        label_path = os.path.join(self.datadir, self.dirnames[index], label_name)
        
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(image_path)
        image = reader.Execute()
        
        if self.load_mod == "img_only":
            # check if clipped:
            img_np = sitk.GetArrayFromImage(image)
            assert np.max(img_np) <= 304 and np.min(img_np) >= -79
            return img_np
        
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(label_path)
        label = reader.Execute()
        
        if self.load_mod == "resample_kits":
            # resample to 3.22*1.62*1.62
            new_spacing = (3.22,1.62,1.62)
            print("resampled data to ", new_spacing)
            image = resampleImage(image, NewSpacing=new_spacing)
            label = resampleImage(label, NewSpacing=new_spacing)

            # clipping 
            new_image = sitk.Clamp(image, lowerBound=-79, upperBound=304)
            new_label = sitk.Clamp(label, lowerBound=-79, upperBound=304)
            return new_image, new_label
        
        if self.load_mod == "all":
            return image, label
        
        if self.load_mod == "np":
            return sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(label)
        
    def z_score(self, index, output_dir, avg, delta):
        image, label = self.__getitem__(index)
        image_scored = (image*1.0 - avg)/delta
        label_scored = (label*1.0 - avg)/delta
        
        image = sitk.GetImageFromArray(image_scored)
        label = sitk.GetImageFromArray(label_scored)
        
        output_image_path = tfilename(output_dir, self.dirnames[index], "imaging.nii.gz")
        output_label_path = tfilename(output_dir, self.dirnames[index], "segmentation.nii.gz")
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_image_path)
        writer.Execute(image)
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_label_path)
        writer.Execute(label)       
        # print("Written in {} and its label".format(output_image_path))
        
    def z_score_dataset(self, output_dir, avg, delta):
        print("Starting Z-socre dataset from ", self.datadir, "To", output_dir)
        self.load_mod = "np"
        for index in tqdm(range(self.__len__())):
            self.z_score(index, output_dir, avg, delta)
    
    def resample_data(self, index, output_dir):
        new_image, new_label = self.__getitem__(index)
        
        output_image_path = tfilename(output_dir, self.dirnames[index], "imaging.nii.gz")
        output_label_path = tfilename(output_dir, self.dirnames[index], "segmentation.nii.gz")
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_image_path)
        writer.Execute(new_image)
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_label_path)
        writer.Execute(new_label)       
        print("Written in {} and its label".format(output_image_path))

    def resample_dataset(self, output_dir):
        print("Starting Resampling dataset from ", self.datadir, "To", output_dir)
        self.load_mod = "resample_kits"
        for index in range(self.__len__()):
            self.resample_data(index, output_dir)
        
    def __len__(self):
        return len(self.dirnames)
    
@tfuncname
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
    
@tfuncname
def test_z_score_1():
    # First: get the avg.
    avg_total = 0
   
    kits = Kits19(load_mod="img_only", datadir="/home1/quanquan/datasets/kits19/resampled_data")
    _len = len(kits)
    for i in tqdm(range(_len)):
        img_np = kits.__getitem__(i)
        avg_total += np.mean(img_np)
    avg = avg_total * 1.0 / _len 
    print(avg)
    np.save("kits-avg.npy", avg)

@tfuncname
def test_z_score_2():
    # Second: get the delta / std
    delta_total = 0
    dim_total = 0
    avg = np.load("kits-avg.npy")
    
    kits = Kits19(load_mod="img_only", datadir="/home1/quanquan/datasets/kits19/resampled_data")
    _len = len(kits)
    for i in tqdm(range(_len)):
        img_np = kits.__getitem__(i)
        # h,w,c = img_np.shape
        # dim = h*w*c
        img_flat = img_np.flatten()
        arr_len = img_flat.shape[0]
        dim_total += arr_len
        for v in img_flat:
            delta_total += (v - avg)**2

    delta = np.sqrt(delta_total*1.0 / dim_total)
    print("delta:", delta)
    np.save("kits-delta.npy", delta)

def test_z_score_3():
    # Third: normalize data
    avg = np.load("kits-avg.npy")
    print("Avg: ", avg)
    delta = np.load("kits-delta.npy")
    print("Detal: ", delta)
    # exit(0)
    kits = Kits19(load_mod="np", datadir="/home1/quanquan/datasets/kits19/resampled_data")
    kits.z_score_dataset("/home1/quanquan/datasets/kits19/normaled_data", avg, delta)
        
def test_z_score_0():
    kits = Kits19(load_mod="np", datadir="/home1/quanquan/datasets/kits19/resampled_data")
    _len = len(kits)
    image, label = kits.__getitem__(5)
    avg = np.mean(image)
    std = np.std(image)
    print(avg, std)

@tfuncname
def test_resample():
    dataset = Kits19(load_mod="resample_kits")
    # transfer
    dataset.resample_dataset("/home1/quanquan/datasets/kits19/resampled_data")
    # image, label = dataset.__getitem__(0)
    # image_np = sitk.GetArrayFromImage(image)
    # print(image_np[0])
    # image_np
    
if __name__ == "__main__":
    # test_z_score_1()
    # test_z_score_2()
    # test_z_score_3()
    test_z_score_0()