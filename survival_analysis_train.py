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
import loss

import copy
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

def get_args():
    opt = argparse.ArgumentParser()
    opt.add_argument('--epochs', help="Num of epochs", default=300, type=int)
    opt.add_argument('--bs', help="Batch Size", default=256, type=int)
    opt.add_argument('--ckpt_dir', help="where to save checkpoints", default="./checkpoints")
    opt.add_argument('--local_rank', default=0, type=int)
    opt.add_argument('--aug', default=0.2, type=float)
    opt.add_argument('--val_pet_path', help="where to val pet dataset", default="/path/to/val/pet")
    opt.add_argument('--val_ct_path', help="where to val ct dataset", default="/path/to/val/ct")
    opt.add_argument('--val_csv_path', help="where to val csv dataset", default="/path/to/val/csv")
    opt.add_argument('--val_pkl_path', help="where to val is_there_a_tumor pkl", default="/path/to/val/ilabel.pkl")
    opt.add_argument('--train_pet_path', help="where to train pet dataset", default="/path/to/train/pet")
    opt.add_argument('--train_ct_path', help="where to train ct dataset", default="/path/to/train/ct")
    opt.add_argument('--train_csv_path', help="where to train csv dataset", default="/path/to/train/csv")
    opt.add_argument('--train_pkl_path', help="where to train is_there_a_tumor pkl", default="/path/to/train/ilabel.pkl")
    opt.add_argument('--pet_pre_path', help="where to pet pretrain vit-b", default="/path/to/ckpt/pet.pth")
    opt.add_argument('--ct_pre_path', help="where to ct pretrain vit-b", default="/path/to/ckpt/ct.pth")
    opt.add_argument('--loss', help="which classification loss to use, FocalLoss | CustomBCELoss", default="CustomBCELoss")
    return opt.parse_args()
    
    
def train(model, critetion, optimizer, train_loader, epoch, device):
    with tqdm(total=len(train_loader)) as pb:
        losses = []
        for _, sample_batched in enumerate(train_loader):
            petimage, ctimage, status, gt, time, _ = sample_batched
            petimage = petimage.to(device)
            ctimage = ctimage.to(device)
            gt = gt.to(device)
            time = time.to(device)
            
            logits = model(ctimage, petimage, status, time, gt)
            # print(logits.shape, conf.shape, sigt.shape)
            total_loss = critetion(time, logits, gt)
            # print(total_loss)
            
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.cpu().data.numpy())
            str_loss = f"{np.mean(losses):.4f}"
            pb.update(1)
            pb.set_postfix(epoch=epoch, lr=f"{optimizer.state_dict()['param_groups'][0]['lr']:.5f}", loss=str_loss)
            
def eval(model, test_loader, device):
    preds = []
    acts = []
    
    for _, sample_batched in  enumerate(test_loader):
        petimage, ctimage, status, gt, time, _ = sample_batched
        petimage = petimage.to(device)
        ctimage = ctimage.to(device)
        gt = gt.to(device)
        time = time.to(device)
        
        logits = model(ctimage, petimage, status, time, gt)
        logits = logits.cpu().data.numpy()
        gt = gt.cpu().data.numpy()
        preds.extend(logits)
        acts.extend(gt)
    
    preds, acts = np.asarray(preds), np.asarray(acts)
    p = accuracy1(torch.from_numpy(preds), torch.from_numpy(acts))
    return p

# these accuracy function not use for calculate c-index, this only use for evaluate which is best in classification, you can try to change these to cindex cal, however, it will SLOW DOWN the program and take more CPU
# Our results were calculated in the survival_analysis_infer.py(we calculate cindex in pkl file separately with lifelines before, for easier open source, we pack it in a single file) , 100 bootstraps, 95% coff  
# cindex cal code borrow from lifelines(https://github.com/CamDavidsonPilon/lifelines)        
def accuracy(pred, act):
    p = F.sigmoid(pred)
    a = act
    p0 = copy.deepcopy(p)
    p0[p0>0.3] = 1
    p0[p0<=0.3] = 0
    q = torch.sum(p0)
    p0 = torch.abs(p0 - a)
    p0 = float(torch.sum(p0) / a.shape[0])
    p = torch.abs(p - a)
    p = float(torch.sum(p) / a.shape[0])
    print("Average bias: {:.3f} | Accuracy: {:.3f} | total: {} | positive: {} | cal: {}".format(p, p0, a.shape[0], torch.sum(a), q))
    return p

def accuracy1(pred, act):
    p = F.sigmoid(pred)
    a = act
    p0 = copy.deepcopy(p)
    p0[p0>0.3] = 1
    p0[p0<=0.3] = 0
    q = torch.sum(p0)
    p0 = torch.abs(p0 - a)
    p0 = float(torch.sum(p0) / a.shape[0])
    p = torch.abs(p - a)
    p = float(torch.sum(p) / a.shape[0])
    print("Average bias: {:.3f} | Accuracy: {:.3f} | total: {} | positive: {} | cal: {}".format(p, p0, a.shape[0], torch.sum(a), q))
    return p

def auc(pred, act):
    p = pred[:, 0]
    a = act[:, 1]
    c = act[:, 0]
    p = p[c!=0]
    a = a[c!=0]
    pos = p[a==1]
    neg = p[a==0]
    q = pos.shape
    
    
def save_checkpoint(state, ckpt_dir, is_best):
    filename = '%s/ckpt.pth' % (ckpt_dir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth', 'best.pth'))

def is_parallel(model):
    '''Return True if model's type is DP or DDP, else False.'''
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def warmup_scheduler(epoch, warmup_epochs=10):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        return 1
    
def main(args):
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)
    device = torch.device("cuda", local_rank)
    if dist.get_rank() == 0:
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
            
    print("Building Engine...")
    model = risk_model(device=device, ct_pre_path=args.ct_pre_path, pet_pre_path=args.pet_pre_path)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)    
    print("Build Done!")

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    criterion = Loss(args.loss)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.03)
    optimizer = optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_scheduler)
    train_dataset = pre_dataset(pet_path=args.train_pet_path, ct_path=args.train_ct_path, csv_path= args.train_csv_path, pkl_path=args.train_pkl_path, aug=args.aug)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.bs,
                              num_workers=4,
                              sampler=train_sampler)
    
    val_dataset = pre_dataset(pet_path=args.val_pet_path, ct_path=args.val_ct_path, csv_path=args.val_csv_path, pkl_path=args.val_pkl_path, mode="val")
    val_loader = DataLoader(val_dataset,
                             batch_size=args.bs,
                             num_workers=8,
                             shuffle=False)

    
    best_prec1 = 0.
    for epoch in range(args.epochs):
        scheduler.step()
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train(model, criterion, optimizer, train_loader, epoch, device)
        model.eval()
        p = eval(model, val_loader, device)
        if dist.get_rank() == 0:
            print('Epoch: {}, Accuracy: {:.3f}'.format(epoch, 1 - p))

            is_best = (1 - p) > best_prec1
            best_prec1 = max((1 - p), best_prec1)
            state_dict = model.module.state_dict() if is_parallel(model) else model.state_dict()
            save_checkpoint({
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                }, args.ckpt_dir, is_best)
        # end of one epoch
    if dist.get_rank() == 0:
        print(f"Finish Training.")
        print(f"Best Prec: {best_prec1:.3f}")
        print(f"Best ckpt save in: {args.ckpt_dir+'/ckpt.best.pth'}")
        
class CustomBCELoss(nn.Module):
    def __init__(self):
        super(CustomBCELoss, self).__init__()

    def forward(self, pred, target):
        return -torch.mean(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.BCELoss = CustomBCELoss()

    def forward(self, inputss, targets):
        inputs = F.sigmoid(inputss)
        BCE_loss = self.BCELoss(inputs, targets)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = BCE_loss * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * loss
        return F_loss.mean()

class newCoxLoss(nn.Module):
    def __init__(self):
        super(newCoxLoss, self).__init__()
    
    def forward(self, survtime, censor, hazard_pred):
        survtime = survtime.reshape(-1)
        i_ = survtime[:, None]
        j_ = survtime[None, :]
        R_mat = i_ <= j_
        R_mat.to(torch.float32).to(hazard_pred)
        theta = hazard_pred.reshape(-1)
        exp_theta = F.sigmoid(theta)
        loss = -torch.mean((torch.log(exp_theta) - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
        return loss

class Loss(nn.Module):
    def __init__(self, losstype, a=1, b=0.2): #cls first, cls loss down then learn cox.
        super(Loss, self).__init__()
        self.focalloss = loss.__dict__[losstype]()
        self.coxloss = newCoxLoss()
        self.a = a
        self.b = b
    
    def forward(self, survtime, inputs, targets):
        return self.a * self.focalloss(F.sigmoid(inputs), targets) + self.b * self.coxloss(survtime, targets, inputs)
        
        
        
if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
