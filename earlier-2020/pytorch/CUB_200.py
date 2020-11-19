from torch.utils.data import Dataset
from PIL import Image
import os


class CUB_200(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(CUB_200, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.classes_file = os.path.join(root, "classes_txt")
        self.image_class_labels_file = os.path.join(root, "image_class_labels.txt")
        self.images_file = os.path.join(root, "images.txt")
        self.train_test_split_file = os.path.join(root, "train_test_split.txt")
        self.bounding_boxes_file = os.path.join(root, "bounding_boxes.txt")

        self._train_ids = []
        self._test_ids = []
        self._image_id_label = {}
        self._train_path_label = []
        self._test_path_label = []

        self._train_test_split()
        self._get_id_to_label()
        self._get_path_label()

    def _train_test_split(self):
        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if label == '1':
                self._train_ids.append(image_id)
            elif label == '0':
                self._test_ids.append(image_id)
            else:
                raise Exception(' label Error! ')

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            self._image_id_label[image_id] = class_id

    def _get_path_label(self):
        for line in open(self.images_file):
            image_id, image_name = line.strip('\n').split()
            label = self._image_id_label[image_id]
            if image_id in self._train_ids:
                self._train_path_label.append((image_name, label))
            else:
                self._test_path_label.append((image_name, label))

    def __getitem__(self, index):
        if self.train:
            image_name, label = self._train_path_label[index]
        else:
            image_name, label = self._test_path_label[index]
        image_path = os.path.join(self.root, 'images', image_name)
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        label = int(label) - 1
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if self.train:
            return len(self._train_ids)
        else:
            return len(self._test_ids)


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    cub200_root = "/media/trans/mnt/data/Stanford datasets/CUB_200_2011/CUB_200_2011"
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    cub = CUB_200(cub200_root, train=True, transform=transform)
    for img, label in cub:
        print(img.size(), label)
        if img.size(0) != 3:
            raise ValueError("????  3333 ")
