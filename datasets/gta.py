#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
from utils import get_label_info, colored_image_to_segmentation
from pathlib import Path


class Gta(Dataset):
    """
    Dataset class for GTA dataset.

    Args:
        data_path (str): Path to the directory containing the input images.
        label_path (str): Path to the directory containing the label images.
        csv_path (str): Path to the CSV file containing label information.
        train_transform (callable): Optional transform to be applied to the input images.

    Attributes:
        data_path (str): Path to the directory containing the input images.
        label_path (str): Path to the directory containing the label images.
        data (list): List of Path objects representing the input images.
        label (list): List of Path objects representing the label images.
        csv_path (str): Path to the CSV file containing label information.
        train_transform (callable): Transform to be applied to the input images during training.

    Methods:
        __getitem__(self, idx): Retrieves the input image and its corresponding label at the given index.
        __len__(self): Returns the total number of samples in the dataset.
    """

    def __init__(self, data_path="datasets/GTA5/images", label_path="datasets/GTA5/labels",
                 csv_path="datasets/class-label.csv",
                 train_transform=None, label_transform=None):

        super(Gta, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.data = list(Path(self.data_path).glob("*.png"))
        self.label = list(Path(self.label_path).glob("*.png"))
        self.csv_path = csv_path
        self.train_transform = train_transform
        self.label_transform = label_transform

    def __getitem__(self, idx):
        element = self.data[idx]
        tmp = element.name
        label = self.label_path + "/" + tmp
        label_path = Path(label)

        label_info = get_label_info(self.csv_path)

        if self.train_transform and self.label_transform:
            label = np.array(self.label_transform(pil_loader(label_path)))
            label = colored_image_to_segmentation(label, label_info)
            return self.train_transform(pil_loader(element)), label

        elif self.train_transform and not self.label_transform:
            label = np.array(pil_loader(label_path))
            label = colored_image_to_segmentation(label, label_info)
            return self.train_transform(pil_loader(element)), label

        elif not self.train_transform and self.label_transform:
            label = np.array(self.label_transform(pil_loader(label_path)))
            label = colored_image_to_segmentation(label, label_info)
            return pil_loader(element), label

        else:
            label = np.array(pil_loader(label_path))
            label = colored_image_to_segmentation(label, label_info)
            return pil_loader(element), label

    def __len__(self):
        return len(self.data)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
