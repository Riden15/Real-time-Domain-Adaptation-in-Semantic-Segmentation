#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import pathlib
from utils import colored_image_to_segmentation, get_label_info
import numpy as np

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CityScapes(Dataset):
    def __init__(self, mode, data_path = "data/Cityscapes/images/", label_path = "data/Cityscapes/gtFine/", csv_path = "class-label.csv", transform=None):
        """
        Initializes a CityScapes dataset object.

        Args:
            mode (str): The mode of the dataset, either "train" or "val".

        Returns:
            None
        """
        super(CityScapes, self).__init__()
        if mode != "train" and mode != "val":
            return -1
        self.mode = mode
        self.data_path = data_path + self.mode
        self.label_path = label_path + self.mode
        self.data = list(pathlib.Path(self.data_path).glob("*/*.png"))
        self.label = list(pathlib.Path(self.label_path).glob("*/*.png"))
        self.csv_path = csv_path
        self.transform = transform

    def __getitem__(self, idx):
        """
        Retrieves the data and labels for a specific index.

        Args:
            idx (int): The index of the data and labels to retrieve.

        Returns:
            tuple: A tuple containing the loaded image, the ground truth color image, and the ground truth label image.
        """
        element = self.data[idx]
        tmp = element.name.split("_")

        gtColor = self.label_path + "/" + tmp[0] + "/" + tmp[0] + "_" + tmp[1] + "_" + tmp[2] + "_gtFine_color.png"
        gtColor_Path = pathlib.Path(gtColor)

        gtLabelTrain = self.label_path + "/" + tmp[0] + "/" + tmp[0] + "_" + tmp[1] + "_" + tmp[
            2] + "_gtFine_labelTrainIds.png"
        gtLabelTrain_Path = pathlib.Path(gtLabelTrain)

        if self.transform:
            label_info = get_label_info(self.csv_path)
            #label = self.transform(pil_loader(gtColor_Path))
            label = np.array(pil_loader(gtColor_Path))
            label = colored_image_to_segmentation(label, label_info)
            
            return self.transform(pil_loader(element)), label
            #return self.transform(pil_loader(element)), self.transform(pil_loader(gtColor_Path))
            # , self.transform(pil_loader(gtLabelTrain_Path))
        else:
            return pil_loader(element), pil_loader(gtColor_Path), pil_loader(gtLabelTrain_Path)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

