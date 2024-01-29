#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from pathlib import Path
import numpy as np
import torch

from utils import colored_image_to_segmentation, get_label_info


class CityScapes(Dataset):

    def __init__(self, mode, data_path="datasets/Cityscapes/images/", label_path="datasets/Cityscapes/gtFine/",
                 csv_path="datasets/class-label.csv", train_transform=None, label_transform=None):
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
        self.label_transform = label_transform

    def __getitem__(self, idx):
        element = self.data[idx]
        tmp = element.name.split("_")

        gtColor = self.label_path + "/" + tmp[0] + "/" + tmp[0] + "_" + tmp[1] + "_" + tmp[2] + "_gtFine_color.png"
        gtColor_Path = Path(gtColor)

        gtLabelTrain = self.label_path + "/" + tmp[0] + "/" + tmp[0] + "_" + tmp[1] + "_" + tmp[
            2] + "_gtFine_labelTrainIds.png"
        gtLabelTrain_Path = Path(gtLabelTrain)

        label_info = get_label_info(self.csv_path)

        if self.train_transform and self.label_transform:
            img = self.train_transform(pil_loader_RGB(element))
            label = self.label_transform(pil_loader_label(gtLabelTrain_Path))
            return img, np.array(label)

            # label_ = np.array(self.label_transform(pil_loader_RGB(gtColor_Path)))
            # label_ = colored_image_to_segmentation(label_, label_info)
            # return self.train_transform(pil_loader_RGB(element)), label

        elif self.train_transform and not self.label_transform:
            # img = self.train_transform(pil_loader_RGB(element))
            # label = pil_loader_label(gtLabelTrain_Path)
            # return img, np.array(label)

            label = np.array(pil_loader_RGB(gtColor_Path))
            label = colored_image_to_segmentation(label, label_info)
            return self.train_transform(pil_loader_RGB(element)), label

        elif not self.train_transform and self.label_transform:
            # img = pil_loader_RGB(element)
            # label = self.label_transform(pil_loader_label(gtLabelTrain_Path))
            # return img, np.array(label)

            label = np.array(self.label_transform(pil_loader_RGB(gtColor_Path)))
            label = colored_image_to_segmentation(label, label_info)
            return pil_loader_RGB(element), label

        else:
            # img = pil_loader_RGB(element)
            # label = pil_loader_label(gtLabelTrain_Path)
            # return img, np.array(label)

            label = np.array(pil_loader_RGB(gtColor_Path))
            label = colored_image_to_segmentation(label, label_info)
            return pil_loader_RGB(element), label

    def __len__(self):
        return len(self.data)


def pil_loader_label(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def pil_loader_RGB(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
