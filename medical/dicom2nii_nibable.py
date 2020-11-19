from __future__ import print_function
import SimpleITK as sitk
import sys
import os

import nibabel
from nibabel.nicom.dicomreaders import read_mosaic_dir
import numpy as np
import pydicom


def test0():
    dicom_path="/home1/quanquan/datasets/lsw/benign_65/fpAML_55/201_1"
    data, affine, b_values, unit_gradients = read_mosaic_dir(dicom_path="/home1/quanquan/datasets/lsw/benign_65/fpAML_55/201_1")
    print(type(data))
    
    files = os.listdir(dicom_path="/home1/quanquan/datasets/lsw/benign_65/fpAML_55/201_1")
    
    
def test1():
    # if len(sys.argv) < 3:
    #     print("Usage DicomSeries Reader <input_dir> <output_file>")
    #     sys.exit(1)
    # print("Reading Dicom dir: ", sys.argv[1])
    
    dicom_path="/home1/quanquan/datasets/lsw/benign_65/fpAML_55/201_1"
    
    reader = sitk.ImageSeriesReader()
    
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    
    reader.ReadImageInformation()
    
    image = reader.Execute()
    nda = sitk.GetArrayFromImage(image)
    print(nda.shape)
    print(type(nda))
    print("--------")
    
    for k in image.GetMetaDataKeys():
        v = image.GetMetaData(k)
        print("({0}) = = \"{1}\"".format(k, v))
    
    
    size = image.GetSize()
    print(type(image))
    print(size)
    print("Image size: ", size[0], size[1], size[2])
    
    # print("Writing Image", sys.argv[2])
    # sitk.WriteImage(image , sys.argv[2])
    
def test2():
    reader = sitk.ImageFileReader()
    dicom_path="/home1/quanquan/datasets/lsw/benign_65/fpAML_55/201_1"
    filenames = os.listdir(dicom_path)    
    reader.SetFileName(os.path.join(dicom_path, filenames[0]))
    reader.LoadPrivateTagsOn()

    reader.ReadImageInformation()

    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        print("({0}) = = \"{1}\"".format(k, v))

    print("Image Size: {0}".format(reader.GetSize()))
    print("Image PixelType: {0}"
        .format(sitk.GetPixelIDValueAsString(reader.GetPixelID())))


if __name__ == "__main__":
    test2()