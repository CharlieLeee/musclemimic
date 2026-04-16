"""Training losses for the CSD→SMPL pipeline."""

from __future__ import annotations

import torch
from roma import rotvec_to_rotmat


def geodesic_loss(pred_aa: torch.Tensor, gt_aa: torch.Tensor) -> torch.Tensor:
    """Geodesic rotation distance on SO(3), averaged over batch/time/joints.

    Parameters
    ----------
    pred_aa, gt_aa : torch.Tensor
        Shape ``(B, T, J*3)`` axis-angle.

    Returns
    -------
    torch.Tensor
        Scalar mean rotation angle (radians).
    """
    pred_r = rotvec_to_rotmat(pred_aa.reshape(-1, 3))
    gt_r = rotvec_to_rotmat(gt_aa.reshape(-1, 3))

    r_rel = pred_r.transpose(-1, -2) @ gt_r
    trace = r_rel[..., 0, 0] + r_rel[..., 1, 1] + r_rel[..., 2, 2]
    cos = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean()


def compute_losses(
    pred_poses: torch.Tensor,
    pred_betas: torch.Tensor,
    pred_trans: torch.Tensor,
    gt_poses: torch.Tensor,
    gt_betas: torch.Tensor,
    gt_trans: torch.Tensor,
    w_pose: float = 5.0,
    w_shape: float = 0.1,
    w_trans: float = 1.0,
    w_smooth: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Weighted sum of pose (geodesic), shape (L2), translation (L1), smoothness.

    Parameters
    ----------
    pred_poses, gt_poses : torch.Tensor
        Shape ``(B, T, 72)`` axis-angle.
    pred_betas, gt_betas : torch.Tensor
        Shape ``(B, n_betas)``.
    pred_trans, gt_trans : torch.Tensor
        Shape ``(B, T, 3)``.
    w_pose, w_shape, w_trans, w_smooth : float
        Loss-term weights.

    Returns
    -------
    dict[str, torch.Tensor]
        Scalar ``loss`` plus the individual components.
    """
    l_pose = geodesic_loss(pred_poses, gt_poses)
    l_shape = (pred_betas - gt_betas).pow(2).mean()
    l_trans = (pred_trans - gt_trans).abs().mean()

    vel = pred_poses[:, 1:] - pred_poses[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    l_smooth = acc.pow(2).mean() if acc.numel() > 0 else torch.zeros((), device=pred_poses.device)

    total = w_pose * l_pose + w_shape * l_shape + w_trans * l_trans + w_smooth * l_smooth

    return {
        "loss": total,
        "L_pose": l_pose,
        "L_shape": l_shape,
        "L_trans": l_trans,
        "L_smooth": l_smooth,
    }
