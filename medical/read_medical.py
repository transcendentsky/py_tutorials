from __future__ import print_function
import SimpleITK as sitk
import sys
import os

import nibabel
from nibabel.nicom.dicomreaders import read_mosaic_dir
import numpy as np
import pydicom
from tutils import *

def test1():
    # ----------------  Read dicom  -------------------
    benign_dir="/home1/quanquan/datasets/lsw/benign_65/fpAML_55/"
    
    dicom_path="/home1/quanquan/datasets/lsw/benign_65/fpAML_55/201_1"
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    
    dicom_image = reader.Execute()
    dicom_np = sitk.GetArrayFromImage(dicom_image)  # numpy
    
    print(type(dicom_np))
    print(dicom_np.shape)
    i = 35
    dicom_slice = dicom_np[35,:,:]
    print("max {}, min {}, avg {} ".format(np.max(dicom_slice), np.min(dicom_slice), np.average(dicom_slice)))
    pimage = (dicom_np[i,:,:]).astype(np.uint8)
    cv2.imwrite(tfilename("output_test/sitk_201_1_dicom_p{}.jpg".format(i)), pimage)
    
    # for k in image.GetMetaDataKeys():
    #     v = image.GetMetaData(k)
    #     print("({0}) = = \"{1}\"".format(k, v))
    
    
    # img = nib.load(example_filename)
    
    # ----------------------  Read Nrrd -----------------------------
    print("Image NRRD! ")
    inputImageFileName = "/home1/quanquan/datasets/lsw/benign_65/fpAML_55/201_1.nrrd"
    
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")
    reader.SetFileName(inputImageFileName)
    nrrd_image = reader.Execute()
    
    # print(nrrd_image)
    
    # writer = sitk.ImageFileWriter()
    # writer.SetFileName(outputImageFileName)
    # writer.Execute(image)
    
    # image_3D = sitk.Image(256, 128, 64, sitk.sitkInt16)
    # image_2D = sitk.Image(64, 64, sitk.sitkFloat32)
    # image_2D = sitk.Image([32,32], sitk.sitkUInt32)
    image_RGB = sitk.Image([128,64], sitk.sitkVectorUInt8, 3)

    # cv2.imwrite(image_RGB, tfilename("output_test/sitk_201_1.jpg"))
    image_array = sitk.GetArrayViewFromImage(nrrd_image)
    print(type(image_array))
    print(image_array.shape)
    for i in range(75):
        pimage = (image_array[i,:,:]*255.0).astype(np.uint8)
        _sum = np.sum(pimage)
        if _sum > 0:
            print("index {}, sum {}, max {}, min {}, avg {}".format(i, _sum, np.max(pimage), np.min(pimage), np.average(pimage)))
            cv2.imwrite(tfilename("output_test/sitk_201_1_label_p{}.jpg".format(i)), pimage)
            
        


if __name__ == "__main__":
    test1()