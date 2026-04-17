"""Render a csd2smpl prediction NPZ as a 3D SMPL skeleton video + frame.

Takes a single ``.pred.npz`` produced by :mod:`csd2smpl.predict`, runs the
predicted SMPL parameters through ``smplx`` to obtain 24 joint positions per
frame, and renders the SMPL kinematic tree as a 3D line skeleton.

Outputs two files next to each other:

* ``<stem>.png`` — frame 0, useful as a lightweight preview committable to git.
* ``<stem>.mp4`` — the full (or truncated) skeleton animation.

Usage
-----
::

    python -m csd2smpl.scripts.visualize_pred \\
        --pred_npz $HOME/csd2smpl/predictions/ACCAD/Female1Running_c3d/C25_-_run_to_walk1_poses.pred.npz \\
        --smpl_dir $HOME/csd2smpl/body_models/smpl \\
        --out_dir  csd2smpl/examples \\
        --max_frames 300 --fps 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


# SMPL-24 kinematic tree: child → parent edges (from the SMPL paper).
# Used to draw the skeleton as a set of line segments between joints.
SMPL24_PARENTS: tuple[int, ...] = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)


def _smpl_forward(
    pred_poses: np.ndarray,
    pred_betas: np.ndarray,
    pred_trans: np.ndarray,
    smpl_dir: Path,
    gender: str = "neutral",
    device: str = "cpu",
) -> np.ndarray:
    """Return 24 SMPL joint positions per frame.

    Parameters
    ----------
    pred_poses : (T, 72) axis-angle SMPL pose.
    pred_betas : (10,) or (T, 10) body shape.
    pred_trans : (T, 3) root translation.
    smpl_dir : directory containing SMPL_NEUTRAL.pkl (etc.).
    """
    try:
        import smplx
    except ImportError as exc:  # pragma: no cover - environment error
        raise SystemExit(
            "smplx is required for visualize_pred. Install with "
            "`pip install 'smplx>=0.1.28'`."
        ) from exc

    dev = torch.device(device)
    body = smplx.create(
        model_path=str(smpl_dir), model_type="smpl", gender=gender,
        batch_size=pred_poses.shape[0],
    ).to(dev)

    poses = torch.as_tensor(pred_poses, dtype=torch.float32, device=dev)
    trans = torch.as_tensor(pred_trans, dtype=torch.float32, device=dev)
    if pred_betas.ndim == 1:
        betas = torch.as_tensor(pred_betas[None, :], dtype=torch.float32, device=dev)
        betas = betas.expand(poses.shape[0], -1)
    else:
        betas = torch.as_tensor(pred_betas, dtype=torch.float32, device=dev)

    with torch.no_grad():
        out = body(
            global_orient=poses[:, :3],
            body_pose=poses[:, 3:],
            betas=betas,
            transl=trans,
        )
    return out.joints[:, :24, :].cpu().numpy()


def _render_frame(ax, joints_t: np.ndarray, bounds: np.ndarray) -> None:
    """Draw one frame of the SMPL skeleton on the given 3D axes."""
    ax.clear()
    xs, ys, zs = joints_t[:, 0], joints_t[:, 1], joints_t[:, 2]
    ax.scatter(xs, ys, zs, s=16, c="#1f77b4", depthshade=False)
    for child, parent in enumerate(SMPL24_PARENTS):
        if parent < 0:
            continue
        ax.plot(
            [joints_t[child, 0], joints_t[parent, 0]],
            [joints_t[child, 1], joints_t[parent, 1]],
            [joints_t[child, 2], joints_t[parent, 2]],
            color="#333333", linewidth=1.5,
        )
    (x0, y0, z0), (x1, y1, z1) = bounds
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    # Tight camera, no title bar — keeps the PNG legible at small sizes.
    ax.view_init(elev=15, azim=-70)


def _compute_bounds(joints: np.ndarray, pad: float = 0.2) -> np.ndarray:
    """Return (2, 3) bounding box with ``pad`` meters of slack on each side."""
    mn = joints.reshape(-1, 3).min(axis=0) - pad
    mx = joints.reshape(-1, 3).max(axis=0) + pad
    return np.stack([mn, mx])


def render(
    pred_npz: Path,
    smpl_dir: Path,
    out_dir: Path,
    max_frames: int,
    fps: int,
    dpi: int,
) -> tuple[Path, Path]:
    """Render one prediction; returns paths to the PNG and MP4 artifacts."""
    try:
        import imageio.v2 as imageio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment error
        raise SystemExit(
            "matplotlib + imageio are required for visualize_pred. Install with "
            "`pip install matplotlib 'imageio[ffmpeg]>=2.34'`."
        ) from exc

    data = np.load(pred_npz, allow_pickle=True)
    poses = data["pred_poses"].astype(np.float32)   # (T, 72)
    betas = data["pred_betas"].astype(np.float32)   # (10,)
    trans = data["pred_trans"].astype(np.float32)   # (T, 3)
    if poses.shape[0] > max_frames:
        poses, trans = poses[:max_frames], trans[:max_frames]
    print(f"Loaded {pred_npz.name}: {poses.shape[0]} frames")

    joints = _smpl_forward(poses, betas, trans, smpl_dir)
    bounds = _compute_bounds(joints)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pred_npz.stem.replace(".pred", "")
    png_path = out_dir / f"{stem}.png"
    mp4_path = out_dir / f"{stem}.mp4"

    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Frame 0 → PNG (small, git-friendly).
    _render_frame(ax, joints[0], bounds)
    fig.savefig(png_path, bbox_inches="tight")
    print(f"Wrote {png_path}")

    # Full sequence → MP4.
    with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=7) as writer:
        for t in range(joints.shape[0]):
            _render_frame(ax, joints[t], bounds)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    print(f"Wrote {mp4_path}  ({joints.shape[0]} frames @ {fps} fps)")
    return png_path, mp4_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_npz", required=True, type=Path)
    parser.add_argument("--smpl_dir", required=True, type=Path)
    parser.add_argument("--out_dir", type=Path, default=Path("csd2smpl/examples"))
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=90)
    args = parser.parse_args()
    render(
        pred_npz=args.pred_npz, smpl_dir=args.smpl_dir, out_dir=args.out_dir,
        max_frames=args.max_frames, fps=args.fps, dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
