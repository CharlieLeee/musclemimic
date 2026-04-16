"""SMPL prediction head: per-frame features → (pose, shape, translation)."""

from __future__ import annotations

import torch
import torch.nn as nn
from roma import rotmat_to_rotvec, special_gramschmidt


class SMPLHead(nn.Module):
    """Decode encoder features into SMPL parameters.

    Output convention
    -----------------
    poses : ``(B, T, n_joints * 3)`` axis-angle
    betas : ``(B, n_betas)`` shape (one prediction per sequence)
    trans : ``(B, T, 3)`` root translation (in the same units as the input)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_joints: int = 24,
        n_betas: int = 10,
    ) -> None:
        super().__init__()
        self.n_joints = n_joints

        self.pose_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_joints * 6),
        )

        self.shape_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, n_betas),
        )

        self.trans_head = nn.Linear(d_model, 3)

    def forward(
        self, feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map features to SMPL parameters.

        Parameters
        ----------
        feat : torch.Tensor
            Shape ``(B, T, d_model)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(poses, betas, trans)`` with shapes
            ``(B, T, n_joints*3)``, ``(B, n_betas)``, ``(B, T, 3)``.
        """
        b, t, _ = feat.shape
        j = self.n_joints

        rot6d = self.pose_head(feat)                         # (B, T, J*6)
        # Zhou et al 6D: first 3 entries = column 0, last 3 = column 1.
        # reshape(..., 2, 3) then transpose puts the two 3-vectors as columns.
        rot6d = rot6d.reshape(b * t * j, 2, 3).transpose(-1, -2)  # (N, 3, 2)
        rotmat = special_gramschmidt(rot6d)                  # (N, 3, 3)
        aa = rotmat_to_rotvec(rotmat).reshape(b, t, j * 3)   # (B, T, J*3)

        betas = self.shape_head(feat.mean(dim=1))            # (B, n_betas)
        trans = self.trans_head(feat)                        # (B, T, 3)
        return aa, betas, trans
