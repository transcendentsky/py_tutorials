# coding: utf-8
# pip install antspyx

import ants
import os
import numpy as np
import SimpleITK as sitk

from scipy.io import loadmat
# annots = loadmat('cars_train_annos.mat')

def test_ants():
    # fixed = ants.image_read( ants.get_ants_data('r16') ).resample_image((64,64),1,0)
    fixed = ants.image_read("/home1/quanquan/datasets/LPBA40/fixed.nii.gz")
    print("-------------")
    # moving = ants.image_read( ants.get_ants_data('r64') ).resample_image((64,64),1,0)

    names = np.array([x.path for x in os.scandir("/home1/quanquan/datasets/LPBA40/train") if x.name.endswith("nii.gz")])
    name = names[0]
    print("name: ", name)
    moving = ants.image_read(name)

    # fixed.plot(overlay=moving, title='Before Registration')
    mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform='Affine' ) # type_of_transform='SyN'
    print(mytx)
    warped_moving = mytx['warpedmovout']
    # fixed.plot(overlay=warped_moving,
    #            title='After Registration')

    mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving,
                                        transformlist=mytx['fwdtransforms'])
    # fixed.plot(filename="mywarpedimage.nii.gz")
    im = mywarpedimage.numpy()
    im = im.transpose((2,1,0))
    print(im.shape)

    print(mytx['fwdtransforms'])
    # import ipdb; ipdb.set_trace()
    annots = loadmat(mytx['fwdtransforms'][0])
    
    writer = sitk.ImageFileWriter()
    writer.SetFileName("mywarpedimage.nii.gz")
    writer.Execute(sitk.GetImageFromArray(im))

    import ipdb; ipdb.set_trace()

def ants_affine(output_dir, fixed_name, moving_name):
    parent, name = os.path.split(moving_name)
    fixed  = ants.image_read(fixed_name)
    moving = ants.image_read(moving_name)
    mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform='Affine' ) # type_of_transform='SyN'
    mname1 = mytx['fwdtransforms'][0]
    mname3 = mytx['invtransforms'][0]
    assert mname1.endswith(".mat")
    mname2 = os.path.join(output_dir, name[:-7]+".fwd.mat")
    mname4 = os.path.join(output_dir, name[:-7]+".inv.mat")
    os.system("cp {} {}".format(mname1, mname2))
    os.system("cp {} {}".format(mname3, mname4))
        
    mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving,
                                        transformlist=mytx['fwdtransforms'])
    im = mywarpedimage.numpy()
    im = im.transpose((2,1,0))
    
    writer = sitk.ImageFileWriter()
    writer.SetFileName(os.path.join(output_dir, name))
    writer.Execute(sitk.GetImageFromArray(im))
    print("Write Image to ", os.path.join(output_dir, name))
    
def label_affine(mat_dir, label_dir, train_dir):
    for x in os.scandir(mat_dir):
        if x.name.endswith("fwd.mat"):
            mat_path = x.path
            parent, name = os.path.split(mat_path)
            prefix = name[:3]
            
            for x in os.scandir(label_dir):
                if x.name.startswith(prefix):
                    label_path = x.path
                    
                    for x in os.scandir(train_dir):
                        if x.name.startswith(prefix):
                            train_path = x.path
                            fixed  = ants.image_read(train_path)
                            moving = ants.image_read(label_path)
                            mywarpedlabel = ants.apply_transforms(fixed=fixed, moving=moving,
                                        transformlist=[mat_path])
                                        
                            im = mywarpedlabel.numpy()
                            im = im.transpose((2,1,0))
                            
                            writer = sitk.ImageFileWriter()
                            writer.SetFileName(os.path.join(label_dir, prefix+".affine.nii.gz"))
                            writer.Execute(sitk.GetImageFromArray(im))
                            print("Write Image to ", os.path.join(label_dir, prefix+".affine.nii.gz"))
                            

def test_affine_images():
    fixed_path = "/home1/quanquan/datasets/LPBA40/fixed.nii.gz"
    
    img_dir = "/home1/quanquan/datasets/LPBA40/test"
    output_dir = "/home1/quanquan/datasets/LPBA40/affined/test"
    
    for x in os.scandir(img_dir):
        if x.name.endswith("nii.gz"):
            ants_affine(output_dir, fixed_path, x.path)
    
    # exit(0)        
    img_dir = "/home1/quanquan/datasets/LPBA40/train"
    output_dir = "/home1/quanquan/datasets/LPBA40/affined/train"
    
    for x in os.scandir(img_dir):
        if x.name.endswith("nii.gz"):
            ants_affine(output_dir, fixed_path, x.path)

                    

if __name__ == "__main__":
    # test_ants()/home1/quanquan/datasets/LPBA40/affined/train 
    label_affine("/home1/quanquan/datasets/LPBA40/affined/test", "/home1/quanquan/datasets/LPBA40/label", "/home1/quanquan/datasets/LPBA40/test")
    label_affine("/home1/quanquan/datasets/LPBA40/affined/train", "/home1/quanquan/datasets/LPBA40/label", "/home1/quanquan/datasets/LPBA40/train")
    
    