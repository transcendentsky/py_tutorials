# coding: utf-8
import ants
import os
import numpy as np
import SimpleITK as sitk
import cv2
from scipy.io import loadmat


# 1. Affine aug (antspy)
# 2. SyN aug (antspy)
# 3. VoxelMorph 

# 2D imgs
def aug_ants_affine(img1, img2, type_of_transform="Affine"):
    """
    type_of_transform: Affine, SyN
    """
    if type(img1) is np.ndarray or type(img1) is not ants.core.ants_image.ANTsImage:
        img1 = ants.from_numpy(img1)
    if type(img2) is np.ndarray or type(img2) is not ants.core.ants_image.ANTsImage:
        img2 = ants.from_numpy(img2)
    fixed = img1; moving = img2
    mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform=type_of_transform)
    warped_moving = mytx['warpedmovout'] # type ANTsImage, shape is like fixed image (500, 542, 3)
    
    # -----------------------
    annots = loadmat(mytx['fwdtransforms'][0])
    print(annots)
    
    mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving,
                                        transformlist=mytx['fwdtransforms'])
    # save image by cv2
    cv2.imwrite("warpedimage.jpg", mywarpedimage.numpy())
    # cv2.imwrite("warpedmoving.jpg", warped_moving.numpy())
    
    import ipdb; ipdb.set_trace()
    cv2.imwrite("warpedmovout.jpg", mytx['warpedmovout'].numpy())
    cv2.imwrite("warpedfixout.jpg", mytx['warpedfixout'].numpy())
    # cv2.imwrite("fwdtransforms.jpg", mytx['fwdtransforms'].numpy())
    # cv2.imwrite("invtransforms.jpg", mytx['invtransforms'].numpy())
    
    
    pass

def test1():
    img_name1 = "/home1/quanquan/code/py_tutorials/medical/corgi1.jpg"
    img_name2 = "/home1/quanquan/code/py_tutorials/medical/QQ3.jpg"
    img1 = ants.image_read(img_name1).numpy()
    img2 = ants.image_read(img_name2).numpy()
    # fixed_numpy = fixed.numpy() # from antsImage to numpy
    # new_img3 = ants.from_numpy(img1.numpy())
    # print(new_img3)
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    # import ipdb; ipdb.set_trace()   
    aug_ants_affine(img1,img2,type_of_transform="Rigid")
    
    
if __name__ == "__main__":
    test1()