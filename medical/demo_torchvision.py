from PIL import Image
from torchvision import transforms as tf
import matplotlib.pyplot as plt


img = Image.open('corgi1.jpg')

img = tf.Resize((256, 256))(img)
size = (224, 224)

trans = {
    "ToTensor": tf.ToTensor(), 
    # Crop
    'RandomCrop': tf.RandomCrop(size),
    'CenterCrop': tf.CenterCrop(size),
    'RandomResizedCrop': tf.RandomResizedCrop(size=size, scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=2),
    # Filp and Rotation
    'RandomRotation': tf.RandomRotation(30),
    'RandomVerFilp': tf.RandomVerticalFlip(p=1),
    'RandomHorFilp': tf.RandomHorizontalFlip(p=1),
    # Transform
    'Normalize': tf.Compose([
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        tf.ToPILImage()
    ]),
    'RandomErasing': tf.Compose([
        tf.ToTensor(),
        tf.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        tf.ToPILImage()
    ]),
    'Pad_5,10,15,20': tf.Pad((5, 10, 15, 20)),
    'ColorJitter_brightness': tf.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
    'ColorJitter_contrast': tf.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
    'ColorJitter_saturation': tf.ColorJitter(brightness=0, contrast=0, saturation=0.5, hue=0),
    'ColorJitter_hue': tf.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5),
    'Grayscale': tf.Grayscale(num_output_channels=1),
    'RandomGrayscale': tf.RandomGrayscale(p=1),
    # 'LinearTransformation': tf.LinearTransformation(transformation_matrix),
    'Affine_degrees': tf.RandomAffine(degrees=30, translate=None, fillcolor=0, scale=None, shear=None),
    'Affine_translate': tf.RandomAffine(degrees=0, translate=(0.2, 0.2), fillcolor=0, scale=None, shear=None),
    'Affine_scale': tf.RandomAffine(degrees=0, translate=None, fillcolor=0, scale=(0.7, 0.7), shear=None),
    'Affine_shear': tf.RandomAffine(degrees=0, translate=None, fillcolor=0, scale=None, shear=(0, 0, 0, 45)),
}


for k, t in trans.items():
    print(k)
    img_ = t(img)
    # print(img.dtype)
    # print(type(img))
    # print(img.shape)
    # plt.title(k)
    # plt.axis('off')
    # plt.imshow(img_)
    # plt.savefig('./tf/%s.jpg' % k, bbox_inches='tight')