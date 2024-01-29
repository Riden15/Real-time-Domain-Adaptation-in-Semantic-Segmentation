#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
from utils import get_label_info, colored_image_to_segmentation
from pathlib import Path
import random


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
                 train_transform=None, label_transform=None, data_aug_transform=None):
        super(Gta, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.data = list(Path(self.data_path).glob("*.png"))
        self.label = list(Path(self.label_path).glob("*.png"))
        self.csv_path = csv_path
        self.train_transform = train_transform
        self.label_transform = label_transform
        self.data_aug_transform = data_aug_transform


    def __getitem__(self, idx):
        """
        Retrieves the input image and its corresponding label at the given index.

        Args:
            idx (int): The index of the image and label to retrieve.

        Returns:
            tuple: A tuple containing the input image and its corresponding label.

        This method first retrieves the image at the given index from the dataset. It then constructs the path to the
        corresponding label image. It reads the label information from the CSV file specified during the
        initialization of the dataset.

        If both `train_transform` and `label_transform` are available, it applies `label_transform` to the label
        image and then converts the colored image to a segmentation map. It then checks if `data_aug_transform` is
        available and a random number is greater than 0.5. If so, it applies `data_aug_transform` to the input image
        before applying `train_transform`. If `data_aug_transform` is not available or the random number is less than
        or equal to 0.5, it applies `train_transform` to the input image directly. It then returns the transformed
        input image and the transformed label as a tuple.

        If `train_transform` is available but `label_transform` is not, it loads the label image directly and
        converts the colored image to a segmentation map. It then applies `train_transform` to the input image in the
        same way as described above and returns the transformed input image and the label as a tuple.

        If `train_transform` is not available but `label_transform` is, it applies `label_transform` to the label
        image, converts the colored image to a segmentation map, and returns the input image and the transformed
        label as a tuple.

        If neither `train_transform` nor `label_transform` is available, it loads the label image directly,
        converts the colored image to a segmentation map, and returns the input image and the label as a tuple.
        """
        element = self.data[idx]
        tmp = element.name
        label = self.label_path + "/" + tmp
        label_path = Path(label)

        label_info = get_label_info(self.csv_path)

        if self.train_transform and self.label_transform:
            element = pil_loader_RGB(element)
            label = np.array(self.label_transform(pil_loader_RGB(label_path)))
            label = colored_image_to_segmentation(label, label_info)

            if random.random() > 0.5 and self.data_aug_transform:
                image = self.train_transform(self.data_aug_transform(element))
            else:
                image = self.train_transform(element)

            return image, label

        elif self.train_transform and not self.label_transform:
            element = pil_loader_RGB(element)
            label = np.array(pil_loader_RGB(label_path))
            label = colored_image_to_segmentation(label, label_info)

            if random.random() > 0.5 and self.data_aug_transform:
                image = self.train_transform(self.data_aug_transform(element))
            else:
                image = self.train_transform(element)

            return image, label

        elif not self.train_transform and self.label_transform:
            label = np.array(self.label_transform(pil_loader_RGB(label_path)))
            label = colored_image_to_segmentation(label, label_info)
            return pil_loader_RGB(element), label

        else:
            label = np.array(pil_loader_RGB(label_path))
            label = colored_image_to_segmentation(label, label_info)
            return pil_loader_RGB(element), label

    def __len__(self):
        return len(self.data)


def pil_loader_RGB(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
