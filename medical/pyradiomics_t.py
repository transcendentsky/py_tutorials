# coding: utf-8
# pip install pyradiomics


import os
import sys
import numpy  as np
import radiomics
# import radiomics.featureextractor as FEE
from radiomics import featureextractor, getTestCase
"/home1/quanquan/datasets/screencopy/43116392_screencopy/Dynamic_c_c"

# def test_nii_slice():
    

def test_radiomics():
    # 文件名
    main_path =  ''
    ori_name = "/home1/quanquan/datasets/screencopy/nii/dynamic/0001.nii.gz" # r'\brain1_image.nrrd'
    lab_name = "/home1/quanquan/datasets/screencopy/nii/dynamic/seg0001.nii.gz" # r'\brain1_label.nrrd'
    para_name = '/home1/quanquan/datasets/screencopy/nii/dynamic/params.yaml'
    
    # 文件全部路径
    ori_path = main_path + ori_name  
    lab_path = main_path + lab_name
    para_path = main_path + para_name
    print("originl path: " + ori_path)
    print("label path: " + lab_path)
    print("parameter path: " + para_path)
    
    # 使用配置文件初始化特征抽取器
    extractor = featureextractor.RadiomicsFeatureExtractor(para_path)
    print ("Extraction parameters:\n\t", extractor.settings)
    # print ("Enabled filters:\n\t", extractor._enabledImagetypes)
    # print ("Enabled features:\n\t", extractor._enabledFeatures)
    
    # 运行
    result = extractor.execute(ori_path,lab_path)  #抽取特征
    print ("Result type:", type(result))  # result is returned in a Python ordered dictionary
    print ("")
    print ("Calculated features")
    for key, value in result.items():  #输出特征
        print ("\t", key, ":", value)
    
if __name__ == "__main__":
    test_radiomics()