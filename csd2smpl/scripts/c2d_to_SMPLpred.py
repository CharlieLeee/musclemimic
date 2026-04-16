import ezc3d
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx



class MarkerToSMPL(nn.Module):
    def __init__(self, n_markers=67, hidden=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_markers * 3, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.pose_head  = nn.Linear(hidden, 72)   # SMPL θ
        self.shape_head = nn.Linear(hidden, 10)   # SMPL β

    def forward(self, x):               # x: (B, N*3)
        h = self.encoder(x)
        return self.pose_head(h), self.shape_head(h)
    


def smpl_forward(pose, shape):
    out = smpl(
        body_pose=pose[:, 3:],
        global_orient=pose[:, :3],
        betas=shape,
        return_verts=True
    )
    return out.joints, out.vertices




def compute_loss(pred_joints, pred_verts, gt_joints_3d, gt_pose=None):
    
    # Joint position loss (main signal)
    loss_joints = F.mse_loss(pred_joints[:, :24], gt_joints_3d)
    
    # Pose regression loss (if GT SMPL params available)
    loss_pose = F.mse_loss(pred_pose, gt_pose) if gt_pose is not None else 0
    
    # Smoothness regularization (temporal)
    loss_smooth = F.mse_loss(pred_pose[1:], pred_pose[:-1])
    
    return loss_joints + 0.1 * loss_pose + 0.01 * loss_smooth


