import os
import numpy as np
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch
import pandas as pd
import cv2
import pickle
import random


class pre_dataset(data.Dataset):
    def __init__(self, pet_path, ct_path, csv_path, pkl_path, mode="trains", aug=0.0):
        super(pre_dataset, self).__init__()
        assert mode in ["trains", "val", "train", "test"]
        self.aug = aug
        self.petpath = pet_path
        self.ctpath = ct_path
        petlist = set(os.listdir(self.petpath))
        ctlist = set(os.listdir(self.ctpath))
        self.filelist = list(petlist & ctlist)
        with open(pkl_path, "rb") as f:
            self.ilabel = pickle.load(f)
        datasets = pd.read_csv(csv_path)
        self.ids = list(datasets['ID'])
        ages = list(datasets['Age'])
        Sex = list(datasets['Sex'])
        tstage = list(datasets['T_stage'])
        nstage = list(datasets['N_stage'])
        mstage = list(datasets['TNM_stage'])
        self.rfs = list(datasets['RFS_time'])
        self.rfss = list(datasets['RFS_status'])
        self.mfs = list(datasets['MFS_time'])
        self.mfss = list(datasets['MFS_status'])
        self.os = list(datasets['OS_time'])
        self.oss = list(datasets['OS_status'])
        self.pfs = list(datasets['PFS_time'])
        self.pfss = list(datasets['PFS_status'])
        self.status_dict={}
        age_status=["baby"] + ["teenager"] + ["adult"] * 4 + ["old"] * 6
        sex = ["female", "male"]
        t_status = ["none", "small", "normal", "large", "huge"]
        n_status = ["none", "less", "normal", "more"]
        tnm_status = ["Stage I", "Stage II", "Stage III", "StageIV"]
        per_status = ["Short period"] * 3 + ["Regular period"] * 4 + ["Long period"] * 40
        for index, id in enumerate(self.ids):
            status = "{} {} {} {} {} {}".format(age_status[int(ages[index]) // 10],
                                                sex[int(Sex[index])],
                                                t_status[int(tstage[index])],
                                                n_status[int(nstage[index])],
                                                tnm_status[int(mstage[index]) - 1],
                                                per_status[int(self.mfs[index]) // 100]
                                                )
            self.status_dict[id] = [status, index]
        self.mode = mode 
        
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        aug_flag=False
        if random.random() < self.aug:
            aug_flag = True
            scale = random.random() * 1 / 2 + 0.5
            scale_size = int(scale * 224)
            
        filename = self.filelist[index]
        id = self.getname(filename)
        while id not in self.status_dict:
            assert id != ""
            id = self.getid(id)
        status, tid = self.status_dict[id]
        censor = torch.Tensor([int(self.mfss[tid])]).to(torch.float32)
        time = torch.Tensor([int(self.mfs[tid])]).to(torch.float32)
        if self.mode == "train":
            img_label = float(self.ilabel[filename])
            censor = censor * 0.5 + img_label * censor * 0.5
        
        petimage = np.load(os.path.join(self.petpath, filename))
        if aug_flag:
            petimage = cv2.resize(petimage, (scale_size, scale_size), interpolation=cv2.INTER_LINEAR)
            petimage = cv2.resize(petimage, (224, 224), interpolation=cv2.INTER_LINEAR)
            petimage = np.fliplr(petimage)
        petimage = torch.from_numpy(petimage).permute(2,0,1)
        normalize(petimage, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    

        ctimage = np.load(os.path.join(self.ctpath, filename))
        if aug_flag:
            petimage = cv2.resize(petimage, (scale_size, scale_size), interpolation=cv2.INTER_LINEAR)
            petimage = cv2.resize(petimage, (224, 224), interpolation=cv2.INTER_LINEAR)
            petimage = np.fliplr(petimage)
        ctimage = torch.from_numpy(ctimage).permute(2,0,1)
        normalize(ctimage, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        return petimage.to(torch.float32), ctimage.to(torch.float32), status, censor, time, filename

    @staticmethod
    def getname(filename):
        return filename.split("_")[0]
    
    @staticmethod
    def getid(id):
        rm = "-" + id.split("-")[-1]
        return id.replace(rm, "")
        
    
    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        petimage, ctimage, status, censor, time = zip(*batch)
        return torch.stack(petimage, 0), torch.stack(ctimage, 0), status, torch.stack(censor, 0), torch.stack(time, 0)
        