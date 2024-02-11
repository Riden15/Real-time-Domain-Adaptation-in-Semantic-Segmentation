#!/usr/bin/python
# -*- encoding: utf-8 -*-
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.transforms.functional as TF

from utils import get_label_info, colored_image_to_segmentation


class Gta(Dataset):
    """
    A class used to represent the GTA dataset

    ...

    Attributes
    ----------
    data_path : str
        a string representing the path to the data
    label_path : str
        a string representing the path to the labels
    csv_path : str
        a string representing the path to the csv file containing class labels
    transformations : bool
        a boolean indicating whether transformations should be applied to the data
    data_augmentation : bool
        a boolean indicating whether data augmentation should be applied to the data
    args : any
        additional arguments

    Methods
    -------
    __getitem__(idx)
        Returns the image and label at the given index after applying necessary transformations and augmentations
    __len__()
        Returns the length of the data
    """

    def __init__(self, data_path="datasets/GTA5/images", label_path="datasets/GTA5/labels",
                 csv_path="datasets/class-label.csv",
                 transformations=False, data_augmentation=False, args=None):
        super(Gta, self).__init__()
        self.args = args
        self.data_path = data_path
        self.label_path = label_path
        self.data = list(Path(self.data_path).glob("*.png"))
        self.label = list(Path(self.label_path).glob("*.png"))
        self.csv_path = csv_path
        self.transformations = transformations
        self.data_augmentation = data_augmentation

    def __getitem__(self, idx):
        """
        Returns the image and label at the given index after applying the necessary transformations and augmentations

        Parameters
        ----------
            idx : int
                Index of the data to be fetched

        Returns
        -------
            image : Tensor
                The transformed/augmented image
            label : array
                The corresponding label
        """

        img_path = self.data[idx]
        label_path = Path(self.label_path + "/" + img_path.name)

        label_info = get_label_info(self.csv_path)
        img = pil_loader_RGB(img_path)
        label = pil_loader_RGB(label_path)

        if self.transformations and self.data_augmentation:
            image, label = transform_with_aug(img, label, self.args)
        elif self.transformations and not self.data_augmentation:
            image, label = transform(img, label, self.args)
        elif not self.transformations and self.data_augmentation:
            image, label = augmentations(img, label, self.args)
        else:
            image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(img)

        label = colored_image_to_segmentation(label, label_info)

        return image, label

    def __len__(self):
        """
        Returns the length of the data

        Returns
        -------
            int
                Length of the data
        """
        return len(self.data)

    def set_augmentation(self, data_augmentation):
        """
        Sets the data augmentation flag

        Parameters
        ----------
            data_augmentation : bool
                The value to be set
        """
        self.data_augmentation = data_augmentation


def transform(image, label, args):
    """
    Applies transformations to the given image and label

    Parameters
    ----------
        image : PIL Image
            The image to be transformed
        label : PIL Image
            The label to be transformed
        args : any
            Additional arguments

    Returns
    -------
        image : Tensor
            The transformed image
        label : array
            The transformed label
    """

    # transform to TV Tensor
    image = v2.ToImage()(image)

    # Resize
    train_resize = v2.Resize((args.crop_height, args.crop_width), antialias=True)
    label_resize = v2.Resize((args.crop_height, args.crop_width), interpolation=v2.InterpolationMode.NEAREST)
    image = train_resize(image)
    label = label_resize(label)

    # scale to [0, 1]
    image = v2.ToDtype(torch.float32, scale=True)(image)

    # Normalize for ImageNet
    image = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    label = np.array(label)
    return image, label


def augmentations(image, label, args):
    """
    Applies augmentations to the given image and label

    Parameters
    ----------
        image : PIL Image
            The image to be augmented
        label : PIL Image
            The label to be augmented
        args : any
            Additional arguments

    Returns
    -------
        image : Tensor
            The augmented image
        label : array
            The augmented label
    """
    # Transform to tensor
    image = v2.ToImage()(image)

    # color jitter (only train image)
    image = v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)(image)

    # random horizontal flip
    image = v2.RandomHorizontalFlip(p=1)(image)
    label = v2.RandomHorizontalFlip(p=1)(label)

    # Random resize crop
    i, j, h, w = v2.RandomCrop.get_params(
        image, output_size=(args.crop_height, args.crop_width))
    image = TF.crop(image, i, j, h, w)
    label = TF.crop(label, i, j, h, w)

    # scale to [0, 1]
    image = v2.ToDtype(torch.float32, scale=True)(image)

    # Normalize for ImageNet
    image = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    label = np.array(label)

    return image, label


def transform_with_aug(image, label, args):
    """
    Applies transformations and augmentations to the given image and label based on a random condition

    Parameters
    ----------
        image : PIL Image
            The image to be transformed/augmented
        label : PIL Image
            The label to be transformed/augmented
        args : any
            Additional arguments

    Returns
    -------
        image : Tensor
            The transformed/augmented image
        label : array
            The transformed/augmented label
    """

    if random.random() > 0.5:
        image, label = transform(image, label, args)
        return image, label
    else:
        image, label = augmentations(image, label, args)
        return image, label


def pil_loader_RGB(path):
    """
    Loads an image from the given path and converts it to RGB

    Parameters
    ----------
        path : Path
            Path to the image

    Returns
    -------
        img : PIL Image
            The loaded image in RGB format
    """

    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
