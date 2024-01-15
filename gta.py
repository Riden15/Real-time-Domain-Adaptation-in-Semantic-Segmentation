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
    def __init__(self):
        super(Gta, self).__init__()
        self.data_path = "GTA5/images"
        self.label_path = "GTA5/labels"
        self.data = list(pathlib.Path(self.data_path).glob("*.png")) # glob takes all the things 
        # that satisfies the regex, in thi case *.png (all the images are in .png)
        self.label = list(pathlib.Path(self.label_path).glob("*.png"))

    def __getitem__(self, idx):
        element = self.data[idx]
        tmp = element.name
        label = self.label_path+"/"+tmp
        label_path = pathlib.Path(label)

        return pil_loader(element), pil_loader(label_path)

    def __len__(self):
        return len(self.data)