
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()
    
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
    def __init__(self, a=1, b=0):
        super(Loss, self).__init__()
        self.focalloss = CustomBCELoss()
        self.coxloss = CoxLoss()
        self.a = a
        self.b = b
    
    def forward(self, survtime, inputs, targets):
        return self.a * self.focalloss(F.sigmoid(inputs), targets) + self.b * self.coxloss(survtime, targets, inputs)