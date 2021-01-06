import matplotlib.pyplot as plt
import matplotlib.patches as patch
import os

# 输出成图片查看得到boundingbox效果
imagedir = r('D:\test_dataset\labelimage')

if ~os.path.exists(r('D:\test_dataset\test_getbbox')):
    os.mkdir(r('D:\test_dataset\test_getbbox'))
for root, _, fnames in sorted(os.walk(imagedir)):
    for fname in sorted(fnames):
        imagepath = os.path.join(root, fname)
        image = misc.imread(imagepath)
        objectnum, bbox = getboundingbox(image)
        ImageID = fname.split('.')[0]
        
        fig,ax = plt.subplots(1)
        ax.imshow(image)
        for box in bbox:
            rect = patch.Rectangle((box[1], box[0]), box[3]-box[1], box[2]-box[0],edgecolor = 'r', linewidth = 1,fill = False)
            ax.add_patch(rect)
        plt.savefig('D:/test_dataset/test_getbbox/'+ImageID+'.png')