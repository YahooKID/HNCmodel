import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from risk_model import risk_model
from util.pre_dataset import pre_dataset
from lifelines.utils import concordance_index as cindex

import copy
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

def get_args():
    opt = argparse.ArgumentParser()
    opt.add_argument('--epochs', help="Num of epochs", default=100, type=int)
    opt.add_argument('--bs', help="Batch Size", default=256, type=int)
    opt.add_argument('--ckpt_path', help="where to save checkpoints", default="/path/to/checkpoint.pth")
    opt.add_argument('--local_rank', default=0, type=int)
    opt.add_argument('--test_pet_path', help="where to test pet dataset", default="/path/to/test/pet")
    opt.add_argument('--test_ct_path', help="where to test ct dataset", default="/path/to/test/ct")
    opt.add_argument('--test_csv_path', help="where to test csv dataset", default="/path/to/test/csv")
    opt.add_argument('--test_pkl_path', help="where to test is_there_a_tumor pkl", default="/path/to/test/ilabel.pkl")
    return opt.parse_args()
    
            
def eval(model, test_loader, device):
    preds = []
    acts = []
    times = []
    filenames = []
    
    for _, sample_batched in  enumerate(test_loader):
        petimage, ctimage, status, gt, time, filename = sample_batched
        petimage = petimage.to(device)
        ctimage = ctimage.to(device)
        gt = gt.to(device)
        time = time.to(device)
        
        logits = model(ctimage, petimage, status, time, gt)
        logits = F.sigmoid(logits)
        logits = logits * F.sigmoid((time // 30 - 36) / 72)
        logits = logits.cpu().data.numpy().reshape(-1)
        gt = gt.cpu().data.numpy().reshape(-1)
        time = time.cpu().data.numpy().reshape(-1)
        preds.extend(logits)
        acts.extend(gt)
        times.extend(time)
        filenames.extend(filename)
        
    
    handle(time, preds, acts, filenames)
    
    
def bootstrap(times, true_labels, predicted_scores, bootstrap_iterations=100):
    cindices = []
    for _ in range(bootstrap_iterations):
        random_indices = np.random.randint(low=0, high=len(true_labels), size=len(true_labels))
        now_true_labels = np.asarray(true_labels)[random_indices]
        now_predicted_scores = np.asarray(predicted_scores)[random_indices]
        now_times = np.asarray(times)[random_indices]
        cindices.append(cindex(now_times, now_predicted_scores, now_true_labels))
    cindices = np.asarray(cindices)
    lower_bound = np.percentile(cindices, 2.5)
    upper_bound = np.percentile(cindices, 97.5)
    exp = np.mean(cindices)
    return exp, lower_bound, upper_bound, cindices
            

def handle(time, pred, act, filenames):
    ref = {}
    for p, a, f, t in zip(pred, act, filenames, time):
        ref[f] = [p, a, t] if f not in ref else [max(p, ref[f][0]), max(a, ref[f][1]), t]
    true_labels = []
    predicted_scores = []
    times = []
    for key in ref:
        tl , ps, ti = ref[key]
        true_labels.append(tl)
        predicted_scores.append(ps)
        times.append(ti)
    exp, l, u, c_index= bootstrap(times, true_labels, predicted_scores)
    print(f"C-index: {exp} [{l}, {u}]")
        
    
        
    

def main(args):
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if dist.get_rank() == 0:
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
            
    print("Building Engine...")
    model = risk_model(device=device)
    
    
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    test_dataset = pre_dataset(pet_path=args.test_pet_path, ct_path=args.test_ct_path, csv_path=args.test_csv_path, pkl_path=args.test_pkl_path, mode="test")
    test_loader = DataLoader(test_dataset,
                             batch_size=args.bs,
                             num_workers=8,
                             shuffle=False)
    eval(model, test_loader, device)
        
        
        
if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
