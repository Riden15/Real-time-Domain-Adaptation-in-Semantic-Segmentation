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

class CityScapes(Dataset):
    def __init__(self, mode): #mode is "train" or "val"
        super(CityScapes, self).__init__()
        if mode != "train" and mode != "val":
            return -1
        self.mode = mode
        self.data_path = "Cityscapes/images/"+self.mode
        self.label_path = "Cityscapes/gtFine/"+self.mode
        self.data = list(pathlib.Path(self.data_path).glob("*/*.png")) #glob finds all the things
         # in the path that satisfies the regex, in this case */*.png (all the images are in png)
        self.label = list(pathlib.Path(self.label_path).glob("*/*.png"))

    def __getitem__(self, idx):
        element = self.data[idx]
        tmp = element.name.split("_") # 0 = city, 1 and 2 are the numbers, 3 is leftImg8bit.png

        gtColor = self.label_path+"/"+tmp[0]+"/"+ tmp[0]+"_"+tmp[1]+"_"+tmp[2]+"_gtFine_color.png"
        gtColor_Path = pathlib.Path(gtColor)

        gtLabelTrain = self.label_path+"/"+tmp[0]+"/"+ tmp[0]+"_"+tmp[1]+"_"+tmp[2]+"_gtFine_labelTrainIds.png"
        gtLabelTrain_Path = pathlib.Path(gtLabelTrain)
        return pil_loader(element), pil_loader(gtColor_Path), pil_loader(gtLabelTrain_Path)

    def __len__(self):
        return len(self.data)