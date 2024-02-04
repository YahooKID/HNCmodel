import os
import numpy as np
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch

import cv2


class petctdataset(data.Dataset):
    def __init__(self, mode):
        super(petctdataset, self).__init__()
        assert mode in ["PET", "CT"]
        self.petpath = "/path/to/train_mae/imgs/PET"
        self.ctpath = "/path/to/train_mae/imgs/CT"
        petlist = set(os.listdir(self.petpath))
        ctlist = set(os.listdir(self.ctpath))
        self.filelist = list(petlist & ctlist)
        self.mode = mode 
        
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        filename = self.filelist[index]
        if self.mode == "PET":
            petimage = np.load(os.path.join(self.petpath, filename))
            petimage = torch.from_numpy(petimage).permute(2,0,1)
            normalize(petimage, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            return petimage.half(), 0
        
        elif self.mode == "CT":
            ctimage = np.load(os.path.join(self.ctpath, filename))
            ctimage = torch.from_numpy(ctimage).permute(2,0,1)
            normalize(ctimage, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            return ctimage.half(), 0
        