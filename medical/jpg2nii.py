# coding :utf-8
# 给 协和 文泰 做图像转化
import SimpleITK as sitk
import numpy as np
import os
import sys
import cv2

data_dir = "/home1/quanquan/datasets/screencopy/43116392_screencopy/"
cate1 = "T1_C"
cate2 = "T2_C"
cate3 = "Dynamic_c"
cate4 = "Dynamic_c_c"

nii_dir = "/home1/quanquan/datasets/screencopy/nii/dynamic/0001.nii.gz"
nii_dir = "/home1/quanquan/datasets/screencopy/nii/T1C/0001.nii.gz"

def test_trans():
    jpg_dir = data_dir + cate3
    # imglist = np.array([x.path for x in os.scandir(jpg_dir) if x.name.endswith('JPG')])
    # imglist.sort()
    imglist = np.array([os.path.join(jpg_dir, str(i) + ".JPG") for i in range(1,25)])
    print(imglist)
    # import ipdb; ipdb.set_trace()
    np_list = []
    for x in imglist:
        reader = sitk.ImageFileReader()
        reader.SetImageIO("JPEGImageIO")
        reader.SetFileName(x)
        jpg_image = reader.Execute()
        jpg_np = sitk.GetArrayFromImage(jpg_image)
        # print(jpg_np.shape)
        jpg_np = cv2.resize(jpg_np, (800,800))
        jpg_np = cv2.cvtColor(jpg_np,cv2.COLOR_BGR2GRAY)
        np_list.append(jpg_np)
        
    res_np = np.stack(np_list, axis=0)
    print(res_np.shape)
    import ipdb; ipdb.set_trace()
    
    res_image = sitk.GetImageFromArray(res_np)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(os.path.join(jpg_dir, "trans_0001.nii.gz"))
    writer.Execute(res_image)
    print("writer wrote: ", os.path.join(jpg_dir, "trans_0001.nii.gz"))
        
    
    
if __name__ == "__main__":
    test_trans()