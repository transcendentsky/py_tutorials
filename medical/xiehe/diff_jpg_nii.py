# coding :utf-8
# 给 协和 文泰 做图像转化
import SimpleITK as sitk
import numpy as np
import os
import sys
import cv2
from tutils import *
p("dsadas")
from eval_criterion import cal_ALL

from tutils import tsitk


data_dir = "/home1/quanquan/datasets/screencopy/43116392_screencopy/"
cate1 = "T1_C"
cate2 = "T2_C"
cate3 = "Dynamic_c"
cate4 = "Dynamic_c_c"

nii_dir1 = "/home1/quanquan/datasets/screencopy/nii/T1C/0001.nii.gz"
nii_dir2 = "/home1/quanquan/datasets/screencopy/nii/dynamic/0001.nii.gz"


def test_jpg_nii():
    filepath_1 = tfilename(data_dir, cate1, "trans_jHZUK.nii.gz")
    filepath_2 = tfilename(nii_dir1)
    filepath_3 = tfilename(data_dir, cate3, "trans_Ko9eR.nii.gz")
    filepath_4 = tfilename(nii_dir2)
    nii1 = sitk.GetArrayFromImage(read(filepath_1, 'nii'))
    nii2 = sitk.GetArrayFromImage(read(filepath_2, 'nii'))
    nii3 = sitk.GetArrayFromImage(read(filepath_3, 'nii'))
    nii4 = sitk.GetArrayFromImage(read(filepath_4, 'nii'))
    
    a,b,c = nii1.shape
    print(nii1.shape, nii2.shape)
    l1_total, l2_total, l3_total,l4_total = 0,0,0,0 
    for i in range(a):
        # nii1's sizes are not consist with nii2
        nii1_slice, nii2_slice = nii1[i,:,:][:,:,np.newaxis], nii2[i,:,:][:,:,np.newaxis]
        nii1_slice = cv2.resize(nii1_slice, (nii2.shape[1], nii2.shape[2]))
        l1, l2, l3, l4 = cal_ALL(nii1_slice, nii2_slice)
        l4 = 0
        l1_total, l2_total, l3_total,l4_total = add_total((l1_total, l2_total, l3_total,l4_total), (l1, l2, l3, l4))
        print("mse:{} cc:{} psnr:{} ssim:{}".format(l1, l2, l3, l4))
    print("avg mse:{} cc:{} psnr:{} ssim:{}".format(l1_total/a, l2_total/a, l3_total/a,l4_total/a ))
    # 31.7406 0.661058 -14.94915
    
    
    a = nii3.shape[0]
    nii4 = nii4.reshape((-1, nii4.shape[-2], nii4.shape[-1]))
    print(nii3.shape, nii4.shape)
    l1_total, l2_total, l3_total,l4_total = 0,0,0,0
    for i in range(a):# nii1's sizes are not consist with nii2
        nii3_slice, nii4_slice = nii3[i,:,:][:,:,np.newaxis], nii4[i,:,:][:,:,np.newaxis]
        nii3_slice = cv2.resize(nii3_slice, (nii4.shape[1], nii4.shape[2]))
        print(nii3_slice.shape, nii4_slice.shape)
        l1, l2, l3, l4 = cal_ALL(nii3_slice, nii4_slice)
        l4 = 0
        l1_total, l2_total, l3_total,l4_total = add_total((l1, l2, l3, l4), (l1_total, l2_total, l3_total,l4_total))
        print("mse:{} cc:{} psnr:{} ssim:{}".format(l1, l2, l3, l4))
    print("avg mse:{} cc:{} psnr:{} ssim:{}".format(l1_total/a, l2_total/a, l3_total/a,l4_total/a ))
    # 18.1512 0.80060 -12.5495

        
type_dict = {"nifti": "NiftiImageIO",
             "nii"  : "NiftiImageIO",  
             "nrrd" : "NrrdImageIO" ,
             "jpg"  : "JPEGImageIO" ,
             "jpeg" : "JPEGImageIO" , 
             "png"  : "PNGImageIO"  ,
             }        
def read(path, mode:str="nifti"):
    mode = mode.lower()
    if mode in type_dict.keys():
        reader = sitk.ImageFileReader()
        reader.SetImageIO(type_dict[mode])
        reader.SetFileName(path)
        image = reader.Execute()
        return image
        
if __name__ == "__main__":
    test_jpg_nii()