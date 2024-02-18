#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
import torch


class CityScapes(Dataset):
    """
    A class used to represent the CityScapes dataset

    ...

    Attributes
    ----------
    mode : str
        a string representing the mode of the dataset (either "train" or "val")
    data_path : str
        a string representing the path to the data
    label_path : str
        a string representing the path to the labels
    transformations : bool
        a boolean indicating whether transformations should be applied to the data
    args : any
        additional arguments

    Methods
    -------
    __getitem__(idx)
        Returns the image and label at the given index after applying necessary transformations
    __len__()
        Returns the length of the data
    """

    def __init__(self, mode, data_path="datasets/Cityscapes/images/", label_path="datasets/Cityscapes/gtFine/",
                 transformations=False, args=None):
        super(CityScapes, self).__init__()
        if mode != "train" and mode != "val":
            return ValueError("Mode must be either train or val")
        self.mode = mode
        self.data_path = data_path + self.mode
        self.label_path = label_path + self.mode
        self.data = list(Path(self.data_path).glob("*/*.png"))
        self.label = list(Path(self.label_path).glob("*/*.png"))
        self.transformations = transformations
        self.args = args

    def __getitem__(self, idx):
        """
        Returns the image and label at the given index after applying necessary transformations

        Parameters
        ----------
            idx : int
                Index of the data to be fetched

        Returns
        -------
            image : Tensor
                The transformed image
            label : Tensor
                The corresponding label
        """
        img_path = self.data[idx]
        tmp = img_path.name.split("_")

        gtColor = self.label_path + "/" + tmp[0] + "/" + tmp[0] + "_" + tmp[1] + "_" + tmp[2] + "_gtFine_color.png"
        gtColor_Path = Path(gtColor)

        gtLabelTrain = self.label_path + "/" + tmp[0] + "/" + tmp[0] + "_" + tmp[1] + "_" + tmp[
            2] + "_gtFine_labelTrainIds.png"
        gtLabelTrain_Path = Path(gtLabelTrain)

        img = pil_loader_RGB(img_path)
        label = pil_loader_label(gtLabelTrain_Path)
        colored_label = pil_loader_RGB(gtColor_Path)

        if self.transformations:
            image, label = transform(img, label, self.args)
        else:
            image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(img)
            label = v2.Compose([v2.ToImage(), v2.ToDtype(torch.int64)])(label)

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
        label : Tensor
            The transformed label
    """
    # transform to TV Tensor
    image = v2.ToImage()(image)
    label = v2.ToImage()(label)

    # Resize
    train_resize = v2.Resize((args.crop_height, args.crop_width), antialias=True)
    label_resize = v2.Resize((args.crop_height, args.crop_width), interpolation=v2.InterpolationMode.NEAREST)
    image = train_resize(image)
    label = label_resize(label)

    # scale to [0, 1]
    image = v2.ToDtype(torch.float32, scale=True)(image)
    label = v2.ToDtype(torch.int64)(label)

    # Normalize for ImageNet
    image = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    return image, label


def pil_loader_label(path):
    """
    Loads an image from the given path and converts it to grayscale

    Parameters
    ----------
        path : Path
            Path to the image

    Returns
    -------
        img : PIL Image
            The loaded image in grayscale format
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def pil_loader_RGB(path):
    """
    Loads an image from the given path and converts it to RGB

    Parameters
    ----------
        path : str
            Path to the image

    Returns
    -------
        img : PIL Image
            The loaded image in RGB format
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
