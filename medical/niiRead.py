import matplotlib

matplotlib.use('TkAgg')

from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import os

# example_filename = '../ADNI_nii/ADNI_002_S_0413_MR_MPR____N3__Scaled_2_Br_20081001114937668_S14782_I118675.nii'
example_filename = 'D:/media/sample/302_1.nii.gz'
# (zyx), 

img = nib.load(example_filename)
print(img)
print(img.header['db_name'])  # 输出头信息

width, height, queue = img.dataobj.shape  # dataobj: data object
print("width height width: ", width, height, queue)

print("OrthoSlicer3D show")
# OrthoSlicer3D(img.dataobj).show()
print("Img.dataobj show")

# num = 1
# for i in range(0, queue, 50):
#     img_arr = img.dataobj[:, i, :]
#     plt.subplot(5, 4, num)
#     plt.imshow(img_arr, cmap='gray')
#     num += 1

img_arr = img.dataobj[:, 222, :]
print("type: ", type(img_arr))
# plt.xlim(0,25)
# plt.margins(0, 1)
plt.imshow(img_arr, cmap='gray')
plt.show()

img_arr = img.dataobj[222, :, :]
plt.imshow(img_arr, cmap='gray')
plt.show()

def read_nii_file1(nii_path):
    '''
    根据nii文件路径读取nii图像
    '''
    nii_image = nib.load(nii_path)
    return nii_image


def nii_one_slice1(image):
    '''
    显示nii image中的其中1张slice
    '''
    image_arr = image.get_data()
    print(type(image_arr))
    print(image_arr.shape)
    # 注意：nibabel读出的image的data的数组顺序为：Width，Height，Channel
    # 将2d数组转置，让plt正常显示
    image_2d = image_arr[:, :, 85].transpose((1, 0))
    plt.imshow(image_2d, cmap='gray')
    plt.show()


import SimpleITK as sitk


def read_nii_file2(img_path):
    '''
    根据nii文件路径读取nii图像
    '''
    nii_image = sitk.ReadImage(img_path)
    return nii_image


def nii_one_slice2(image):
    '''
    显示nii image中的其中1张slice
    '''
    # C,H,W
    # SimpleITK读出的image的data的数组顺序为：Channel,Height，Width
    image_arr = sitk.GetArrayFromImage(image)
    print(type(image_arr))
    print(image_arr.shape)
    image_2d = image_arr[85, :, :]
    plt.imshow(image_2d, cmap='gray')
    plt.show()


if __name__ == "__main__":
    NII_DIR = 'D:/media/sample/302_1.nii.gz'

    nii_image1 = read_nii_file1(os.path.join(NII_DIR, 'Brats18_2013_2_1', 'Brats18_2013_2_1_flair.nii.gz'))
    nii_one_slice1(nii_image1)

    nii_image2 = read_nii_file2(os.path.join(NII_DIR, 'Brats18_2013_2_1', 'Brats18_2013_2_1_flair.nii.gz'))
    nii_one_slice2(nii_image2)
