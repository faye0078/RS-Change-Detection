import numpy as np
import os
from paddle.io import Dataset
from utils.preprocess import Compose
from paddleseg.transforms import Resize

class ConcatDataset(Dataset):
    def __init__(self, root_dir, path, num_classes=None, transforms=None):
        super(ConcatDataset, self).__init__()

        self.root_dir = root_dir
        self.transforms = Compose(transforms, concat=True)
        with open(path, "rb") as f:
            datalist = f.readlines()
        try:
            self.datalist = [
                (k[0], k[1], k[2])
                for k in map(
                    lambda x: x.decode("utf-8").strip("\n").strip("\r").split("\t"), datalist
                )
            ]
        except ValueError:
            print("Error: wrong file_list")

        self.data_list = datalist
        self.data_num = len(self.data_list)

    def __getitem__(self, idx):
        im1_name = os.path.join(self.root_dir, self.datalist[idx][0])
        im2_name = os.path.join(self.root_dir, self.datalist[idx][1])
        lab_name = os.path.join(self.root_dir, self.datalist[idx][2])

        img, label = self.transforms(im1_name, im2_name, lab_name)

        return (img, label)

    def __len__(self):
        return self.data_num


class SplitDataset(Dataset):
    def __init__(self, root_dir, path, num_classes=None, transforms=None):
        super(SplitDataset, self).__init__()

        self.root_dir = root_dir
        self.transforms = Compose(transforms, concat=False)
        with open(path, "rb") as f:
            datalist = f.readlines()
        try:
            self.datalist = [
                (k[0], k[1], k[2])
                for k in map(
                    lambda x: x.decode("utf-8").strip("\n").strip("\r").split("\t"), datalist
                )
            ]
        except ValueError:
            print("Error: wrong file_list")

        self.data_list = datalist
        self.data_num = len(self.data_list)

    def __getitem__(self, idx):
        im1_name = os.path.join(self.root_dir, self.datalist[idx][0])
        im2_name = os.path.join(self.root_dir, self.datalist[idx][1])
        lab_name = os.path.join(self.root_dir, self.datalist[idx][2])

        im1, im2, label = self.transforms(im1_name, im2_name, lab_name)

        return (im1, im2, label)

    def __len__(self):
        return self.data_num

        

