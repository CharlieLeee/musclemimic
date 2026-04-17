"""Evaluation: MPJPE, PA-MPJPE, V2V on a held-out split.

Requires ``smplx`` and SMPL body-model files (.pkl) to decode predicted
parameters back into joints/vertices.

Usage
-----
::

    python -m csd2smpl.evaluate \\
        --config   csd2smpl/configs/default.yaml \\
        --ckpt     checkpoints/best.pt \\
        --split    test \\
        --smpl_dir body_models/smpl
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from csd2smpl.data.c3d_dataset import MarkerDataset
from csd2smpl.data.synthesize import _resolve_smplx_model_path
from csd2smpl.models.pipeline import Markers2SMPL


def mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> float:
    """Mean Per-Joint Position Error in millimetres.

    Parameters
    ----------
    pred_joints, gt_joints : torch.Tensor
        Shape ``(B, T, J, 3)``, metres.
    """
    return (pred_joints - gt_joints).norm(dim=-1).mean().item() * 1000


def procrustes_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Rigid (translation + rotation) Procrustes alignment of ``pred`` onto ``gt``.

    Parameters
    ----------
    pred, gt : torch.Tensor
        Shape ``(N, 3)``.

    Returns
    -------
    torch.Tensor
        ``pred`` aligned to ``gt``, shape ``(N, 3)``.
    """
    mu_p = pred.mean(0, keepdim=True)
    mu_g = gt.mean(0, keepdim=True)
    pred_c = pred - mu_p
    gt_c = gt - mu_g
    u, _, vt = torch.linalg.svd(gt_c.T @ pred_c)
    r = u @ vt
    return (pred_c @ r.T) + mu_g


def pa_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> float:
    """Procrustes-aligned MPJPE (mm)."""
    b, t, j, _ = pred_joints.shape
    aligned: list[torch.Tensor] = []
    for bi in range(b):
        for ti in range(t):
            aligned.append(
                procrustes_align(pred_joints[bi, ti], gt_joints[bi, ti])
            )
    aligned_tensor = torch.stack(aligned).reshape(b, t, j, 3)
    return mpjpe(aligned_tensor, gt_joints)


def evaluate(cfg: dict, ckpt_path: str, split: str, smpl_dir: str) -> None:
    """Decode predictions through SMPL and report MPJPE / PA-MPJPE / V2V."""
    import smplx  # imported lazily — keep evaluate.py importable without smplx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MarkerDataset(
        cfg["data_root"], split=split,
        seq_len=cfg["seq_len"], stride=cfg["seq_len"],
    )
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=2,
    )

    model = Markers2SMPL(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    smpl = smplx.create(
        _resolve_smplx_model_path(smpl_dir),
        model_type="smpl", num_betas=10,
        batch_size=cfg["batch_size"] * cfg["seq_len"],
    ).to(device)

    def decode(poses: torch.Tensor, betas: torch.Tensor, trans: torch.Tensor):
        b, t, _ = poses.shape
        flat_poses = poses.reshape(b * t, -1)
        flat_betas = betas.unsqueeze(1).expand(-1, t, -1).reshape(b * t, -1)
        flat_trans = trans.reshape(b * t, 3)
        out = smpl(
            global_orient=flat_poses[:, :3],
            body_pose=flat_poses[:, 3:],
            betas=flat_betas,
            transl=flat_trans,
        )
        return out.joints[:, :24].reshape(b, t, 24, 3), out.vertices.reshape(b, t, -1, 3)

    all_mpjpe: list[float] = []
    all_pa: list[float] = []
    all_v2v: list[float] = []

    with torch.no_grad():
        for batch in loader:
            markers = batch["markers"].to(device)
            mask = batch["mask"].to(device)
            gt_poses = batch["poses_gt"].to(device)
            gt_betas = batch["betas_gt"].to(device)
            gt_trans = batch["trans_gt"].to(device)

            pred_poses, pred_betas, pred_trans = model(markers, mask)

            pred_j, pred_v = decode(pred_poses, pred_betas, pred_trans)
            gt_j, gt_v = decode(gt_poses, gt_betas, gt_trans)

            all_mpjpe.append(mpjpe(pred_j, gt_j))
            all_pa.append(pa_mpjpe(pred_j, gt_j))
            all_v2v.append((pred_v - gt_v).norm(dim=-1).mean().item() * 1000)

    print(f"\n── Evaluation [{split}] ──────────────────────")
    print(f"  MPJPE     : {np.mean(all_mpjpe):.1f} mm")
    print(f"  PA-MPJPE  : {np.mean(all_pa):.1f} mm")
    print(f"  V2V       : {np.mean(all_v2v):.1f} mm")
    print("─────────────────────────────────────────────\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="csd2smpl/configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--smpl_dir", default="body_models/smpl")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg, args.ckpt, args.split, args.smpl_dir)


if __name__ == "__main__":
    main()
