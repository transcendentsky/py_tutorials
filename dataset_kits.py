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
        print("Starting Resample dataset from")
        print(self.datadir)
        print("To")
        print(output_dir)
        self.load_mod = "resample_kits"
        for index in range(self.__len__()):
            self.resample_data(index, output_dir)
            pass
        
    def __len__(self):
        return len(self.dirnames)
    

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
    dataset = Kits19(load_mod="resample_kits")
    # transfer
    dataset.resample_dataset("/home1/quanquan/datasets/kits19/resampled_data")
    # image, label = dataset.__getitem__(0)
    # image_np = sitk.GetArrayFromImage(image)
    # print(image_np[0])
    # image_np