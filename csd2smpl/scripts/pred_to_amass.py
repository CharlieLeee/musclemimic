"""Repack a csd2smpl prediction NPZ into the AMASS stageii schema.

The musclemimic retargeter (``loco_mujoco/smpl/retargeting.py::load_amass_data``)
globs every ``*.npz`` under ``AMASS_PATH`` and looks up motions by relative
path. It reads a fixed set of keys: ``poses`` (at least 66 axis-angle dims,
SMPL-H body), ``trans``, ``betas``, ``gender``, ``mocap_framerate``.

Our ``.pred.npz`` has ``pred_poses (T, 72)``, ``pred_betas (10,)``,
``pred_trans (T, 3)``. Mapping:

* ``pred_poses`` → ``poses`` (passthrough; the loader slices ``[:, :66]``).
* ``pred_trans`` → ``trans``.
* ``pred_betas`` → ``betas``.
* ``gender``     → "neutral" (our training data isn't gender-conditioned).
* ``fps``        → ``mocap_framerate``.

Usage
-----
::

    python -m csd2smpl.scripts.pred_to_amass \\
        --pred_npz   $HOME/csd2smpl/predictions/ACCAD/.../seq.pred.npz \\
        --amass_root $HOME/csd2smpl/amass \\
        --subset     CsdPred \\
        --motion     seq_01 \\
        --fps        30

This writes ``$amass_root/CsdPred/seq_01_poses.npz`` with the AMASS schema,
so the musclemimic viewer can address it as ``CsdPred/seq_01_poses``::

    AMASS_PATH=$HOME/csd2smpl/amass \\
    python examples/retargeting/retarget_visualize.py \\
        --motion CsdPred/seq_01_poses --record
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def repack(
    pred_npz: Path,
    amass_root: Path,
    subset: str,
    motion: str,
    fps: float,
    gender: str,
) -> Path:
    """Write an AMASS-schema NPZ and return its path."""
    data = np.load(pred_npz, allow_pickle=True)
    missing = {"pred_poses", "pred_betas", "pred_trans"} - set(data.files)
    if missing:
        raise ValueError(
            f"{pred_npz} is not a csd2smpl prediction NPZ "
            f"(missing keys: {sorted(missing)}, got {sorted(data.files)})"
        )

    poses = data["pred_poses"].astype(np.float32)  # (T, 72)
    trans = data["pred_trans"].astype(np.float32)  # (T, 3)
    betas = data["pred_betas"].astype(np.float32)  # (10,) or (T, 10)
    if betas.ndim == 2:
        # The loader uses a single (1, N) beta vector; collapse to the mean.
        betas = betas.mean(axis=0)

    if poses.shape[0] != trans.shape[0]:
        raise ValueError(
            f"frame-count mismatch: poses={poses.shape[0]} trans={trans.shape[0]}"
        )
    if poses.shape[1] < 66:
        raise ValueError(
            f"expected ≥66 pose dims (SMPL-H body); got {poses.shape[1]}"
        )

    out_dir = amass_root / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{motion}_poses.npz"

    np.savez(
        out_path,
        poses=poses,
        trans=trans,
        betas=betas,
        gender=np.array(gender),                # stored as 0-d string array
        mocap_framerate=np.float32(fps),
    )
    print(
        f"Wrote {out_path}\n"
        f"  poses={poses.shape}  trans={trans.shape}  "
        f"betas={betas.shape}  fps={fps}  gender={gender}\n"
        f"Reference this motion in musclemimic as: {subset}/{motion}_poses"
    )
    return out_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_npz", required=True, type=Path)
    parser.add_argument("--amass_root", required=True, type=Path)
    parser.add_argument(
        "--subset", default="CsdPred",
        help="sub-dataset directory under amass_root (default: CsdPred)",
    )
    parser.add_argument(
        "--motion", default=None,
        help="motion stem (defaults to the pred file's stem, minus .pred)",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--gender", default="neutral")
    args = parser.parse_args()

    motion = args.motion or args.pred_npz.stem.replace(".pred", "")
    repack(
        pred_npz=args.pred_npz, amass_root=args.amass_root,
        subset=args.subset, motion=motion, fps=args.fps, gender=args.gender,
    )


if __name__ == "__main__":
    main()
