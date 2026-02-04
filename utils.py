import torch
import torch.nn as nn
import torch.nn.functional as F
from copent import transent
import numpy as np


class EgoLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, pred, gt, mask=None):
        # pred: [B, T, 2], gt: [B, T, 2], mask: [B, T]
        loss = F.smooth_l1_loss(pred, gt, reduction='none')  # [B, T, 2]
        loss = loss.sum(dim=-1)  # [B, T]
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        else:
            return loss.mean()


class CTEDistillLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_cte, true_cte, mask=None):
        loss = (pred_cte - true_cte) ** 2
        if mask is not None:
            loss = loss * mask
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def _to_speed_np(xy):
    return np.diff(xy,axis=0)

def compute_cte(ego_traj, cand_trajs, lag=1, k=5, dt_type=2, use_velocity=True):
    device = ego_traj.device
    ego_np = ego_traj.squeeze(0).detach().cpu().numpy()  # [T,2]
    if use_velocity:
        ego_np = _to_speed_np(ego_np)
    scores = []
    for i in range(cand_trajs.shape[0]):
        c = cand_trajs[i].detach().cpu().numpy()
        if use_velocity:
            c = _to_speed_np(c)
        Tm = min(len(ego_np), len(c))
        if Tm <= lag or Tm < 2:
            scores.append(0.0); continue
        try:
            s = float(transent(c[:Tm], ego_np[:Tm], lag=lag, k=k))
        except Exception:
            s = 0.0
        scores.append(s)
    return torch.tensor(scores, dtype=torch.float32, device=device)
