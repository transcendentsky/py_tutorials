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

# Dataset Structure (nnUNet Style)
# /path/to/decathlon/
#     Task01_BrainTumor/
#         imagesTr/
#             BRATS_001.nii.gz
#             BRATS_xxx.nii.gz
#             ......
#         imagesTs/
#             BRATS_485.nii.gz
#             ......
#         labelsTr/
#             BRATS_000.nii.gz
#             ......
#     Task02_Heart/
#     Task03_Liver/
#     ......
#     ......

class Decathon(Dataset):
    def __init__(self, load_mod:str="all", datadir:str="/home1/quanquan/datasets/decathlon", task_id="01", train="train"):
        self.datadir  = datadir
        self.load_mod = load_mod
        self.task_id = task_id
        self.tasks = np.array([x.name for x in os.scandir(datadir) if x.name.startswith("Task")])
        print(f"Tasks: {self.tasks}")
        if task_id != "all" and task_id is not None:
            self.task_name = self.get_task_name(task_id)
            if train == "train":
                self.subdir_name = "imagesTr"
            elif train == "test":
                self.subdir_name = "imagesTs"
            self.files = np.array([x.name for x in os.scandir(os.path.join(self.datadir, self.task_name, self.subdir_name)) if x.name.endswith("nii.gz") and not x.name.startswith(".")])
            self.files.sort()
            
        elif task_id == "all":
            raise NotImplementedError
        else:
            raise ValueError("Task ID Error!")
        
    def get_task_name(self, task_id):
        for task in self.tasks:
            if task.startswith("Task"+task_id):
                d("Find ", task, "Task"+task_id)
                return task
        raise ValueError("Task ID Error!")
        return None
    
    def __getitem__(self, index:int):
        filename = self.files[index]
        d("filename:", filename)
        image_path = os.path.join(self.datadir, self.task_name, self.subdir_name, filename)
        label_path = os.path.join(self.datadir, self.task_name, "labelsTr", filename)
        
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(image_path)
        image = reader.Execute()
        
        if self.load_mod == "img_only":
            # check if clipped:
            img_np = sitk.GetArrayFromImage(image)
            assert np.max(img_np) <= 304 and np.min(img_np) >= -79
            return img_np
        elif self.load_mod == "check_img_only":
            return image
        
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(label_path)
        label = reader.Execute()
        
        raise NotImplementedError
    
    def check_property(self):
        # spacing, sizes
        self.load_mod = "check_img_only"
        print("[DEBUG] change Load Mod to 'check_img_only'. ")
        for index in range(self.__len__()):
            image = self.__getitem__(index)
            size = image.GetSize()
            spacing = image.GetSpacing()
            print(f"spacing: {spacing}")
        
    def __len__(self):
        if self.task_id != "all" and self.task_id is not None:
            return len(self.files)
        
        
def test_check_data_spacing():
    dataset = Decathon(task_id="10")
    dataset.check_property()
    
if __name__ == "__main__":
    test_check_data_spacing()