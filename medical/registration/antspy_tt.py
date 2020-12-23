import ants
import os
import numpy as np

# fixed = ants.image_read( ants.get_ants_data('r16') ).resample_image((64,64),1,0)
fixed = ants.image_read("/home1/quanquan/datasets/LPBA40/fixed.nii.gz")
print("-------------")
# moving = ants.image_read( ants.get_ants_data('r64') ).resample_image((64,64),1,0)

names = np.array([x.path for x in os.scandir("/home1/quanquan/datasets/LPBA40/train") if x.name.endswith("nii.gz")])
name = names[0]
moving = ants.image_read(name)

# fixed.plot(overlay=moving, title='Before Registration')
mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform='SyN' )
print(mytx)
warped_moving = mytx['warpedmovout']
# fixed.plot(overlay=warped_moving,
#            title='After Registration')