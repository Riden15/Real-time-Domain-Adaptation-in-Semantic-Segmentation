#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import pathlib


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Gta(Dataset):
    """
    Dataset class for GTA5 dataset.

    Attributes:
        data_path (str): Path to the directory containing the images.
        label_path (str): Path to the directory containing the labels.
        data (list): List of image file paths.
        label (list): List of label file paths.
    """

    def __init__(self, transform=None):
        super(Gta, self).__init__()
        self.data_path = "data/GTA5/images"
        self.label_path = "data/GTA5/labels"
        self.data = list(pathlib.Path(self.data_path).glob("*.png"))  # glob takes all the things
        # that satisfies the regex, in thi case *.png (all the images are in .png)
        self.label = list(pathlib.Path(self.label_path).glob("*.png"))
        self.transform = transform

    def __getitem__(self, idx):
        """
        Get the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the loaded image and label.
        """
        element = self.data[idx]
        tmp = element.name
        label = self.label_path + "/" + tmp
        label_path = pathlib.Path(label)

        if self.transform:
            return self.transform(pil_loader(element)), self.transform(pil_loader(label_path))
        else:
            return pil_loader(element), pil_loader(label_path)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)
