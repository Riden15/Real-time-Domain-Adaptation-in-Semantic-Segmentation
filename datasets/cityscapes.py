#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os

from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision.transforms import transforms
from utils import colored_image_to_segmentation, get_label_info
import numpy as np


class CityScapes(Dataset):
    """
    Dataset class for CityScapes dataset.

    Args:
        mode (str): The mode of the dataset, either "train" or "val".
        data_path (str): The path to the directory containing the image data.
        label_path (str): The path to the directory containing the label data.
        csv_path (str): The path to the CSV file containing label information.
        train_transform (callable): Optional transform to be applied to the image data during training.

    Attributes:
        mode (str): The mode of the dataset, either "train" or "val".
        data_path (str): The path to the directory containing the image data.
        label_path (str): The path to the directory containing the label data.
        data (list): A list of Path objects representing the image files.
        label (list): A list of Path objects representing the label files.
        csv_path (str): The path to the CSV file containing label information.
        train_transform (callable): Optional transform to be applied to the image data during training.

    Methods:
        __getitem__(self, idx): Retrieve the image and label at the given index.
        __len__(self): Returns the total number of images in the dataset.
    """

    def __init__(self, mode, data_path="datasets/Cityscapes/images/", label_path="datasets/Cityscapes/gtFine/",
                 csv_path="datasets/class-label.csv", train_transform=None):
        super(CityScapes, self).__init__()
        if mode != "train" and mode != "val":
            return -1
        self.mode = mode
        self.data_path = data_path + self.mode
        self.label_path = label_path + self.mode
        self.data = list(Path(self.data_path).glob("*/*.png"))
        self.label = list(Path(self.label_path).glob("*/*.png"))
        self.csv_path = csv_path
        self.train_transform = train_transform

    def __getitem__(self, idx):
        element = self.data[idx]
        tmp = element.name.split("_")

        gtColor = self.label_path + "/" + tmp[0] + "/" + tmp[0] + "_" + tmp[1] + "_" + tmp[2] + "_gtFine_color.png"
        gtColor_Path = Path(gtColor)

        gtLabelTrain = self.label_path + "/" + tmp[0] + "/" + tmp[0] + "_" + tmp[1] + "_" + tmp[
            2] + "_gtFine_labelTrainIds.png"
        gtLabelTrain_Path = Path(gtLabelTrain)

        # obtaining semantic segmentation labels from image
        label_info = get_label_info(self.csv_path)
        # label = self.transform(pil_loader(gtColor_Path))
        label = np.array(pil_loader(gtColor_Path))
        label = colored_image_to_segmentation(label, label_info)

        if self.train_transform:
            return self.train_transform(pil_loader(element)), label
        else:
            return pil_loader(element), label

    def __len__(self):
        return len(self.data)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
