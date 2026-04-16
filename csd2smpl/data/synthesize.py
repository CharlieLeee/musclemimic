"""Offline synthesis: AMASS NPZ → markers + SMPL targets NPZ.

AMASS schema (Mahmood et al. 2019, confirmed against
musclemimic/loco_mujoco/smpl/retargeting.py)::

    poses             (T, 156)  axis-angle, SMPL-H layout:
                                    poses[:, :3]      global orientation
                                    poses[:, 3:66]    body pose (21 joints × 3)
                                    poses[:, 66:111]  L-hand pose (15 joints × 3)
                                    poses[:, 111:156] R-hand pose (15 joints × 3)
    betas             (16,)     shape coefficients (SMPL-H beta basis)
    trans             (T, 3)    root translation (metres)
    gender            scalar    'male' / 'female' / 'neutral'
    mocap_framerate   scalar    Hz (older subsets); newer subsets use mocap_frame_rate

Output NPZ (one per AMASS sequence) — this is the dataset contract::

    markers      (T_out, M, 3)   float32   — synthetic mocap marker positions (m)
    markers_clean(T_out, M, 3)   float32   — pre-noise markers, for debugging
    mask         (T_out, M)      bool      — True = marker valid
    poses        (T_out, 72)     float32   — SMPL-24 axis-angle (supervision)
    betas        (10,)           float32   — SMPL-10 shape (supervision)
    trans        (T_out, 3)      float32   — root translation (supervision)
    layout       ()              str       — marker-layout name (e.g. "cmu_41")
    marker_names ()              str       — comma-joined marker labels
    gender       ()              str       — 'male' / 'female' / 'neutral'
    fps          ()              float32   — output framerate
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from csd2smpl.data.marker_layouts import (
    LAYOUTS,
    MarkerSpec,
    get_layout,
    layout_to_arrays,
)


def amass_to_smpl72(poses_amass: np.ndarray) -> np.ndarray:
    """Convert AMASS SMPL-H poses ``(T, 156)`` to SMPL-24 axis-angle ``(T, 72)``.

    SMPL has 24 joints; SMPL-H replaces SMPL's last two body joints (l_hand,
    r_hand at indices 22, 23) with full hand articulation. To collapse back to
    SMPL-24 we keep the first 22 SMPL-H joints (66 dims) and zero-pad two more.
    """
    if poses_amass.shape[1] < 66:
        raise ValueError(
            f"AMASS poses must have >=66 columns (SMPL-H body); got {poses_amass.shape}"
        )
    body_22 = poses_amass[:, :66].astype(np.float32)
    pad = np.zeros((body_22.shape[0], 6), dtype=np.float32)
    return np.concatenate([body_22, pad], axis=-1)


def get_amass_framerate(seq) -> float:
    """Return the framerate (Hz) of an AMASS NPZ, normalising key spellings."""
    keys = seq.files if hasattr(seq, "files") else list(seq.keys())
    for key in ("mocap_framerate", "mocap_frame_rate"):
        if key in keys:
            return float(seq[key])
    raise KeyError("AMASS NPZ has no 'mocap_framerate' or 'mocap_frame_rate'")


def downsample(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Stride-based downsampling along axis 0. No interpolation."""
    if dst_fps <= 0 or src_fps <= dst_fps:
        return arr
    step = max(1, int(round(src_fps / dst_fps)))
    return arr[::step]


def smpl_forward_joints(
    poses_smpl72: np.ndarray,
    betas10: np.ndarray,
    trans: np.ndarray,
    model_path: str,
    gender: str = "neutral",
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Run SMPL forward kinematics and return the 24 SMPL joint world positions.

    Parameters
    ----------
    poses_smpl72 : np.ndarray
        Shape ``(T, 72)`` axis-angle (SMPL body, hand-joints zero-padded).
    betas10 : np.ndarray
        Shape ``(10,)``.
    trans : np.ndarray
        Shape ``(T, 3)``.
    model_path : str
        Directory containing SMPL_NEUTRAL/MALE/FEMALE.pkl.
    gender : str
        'neutral' | 'male' | 'female'.
    batch_size : int
        Frames per smplx forward call.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    np.ndarray
        Shape ``(T, 24, 3)``, metres.
    """
    import smplx  # imported lazily so the package is inspectable without it

    body_model = smplx.create(
        model_path,
        model_type="smpl",
        gender=gender,
        num_betas=10,
        batch_size=batch_size,
    ).to(device)

    t = poses_smpl72.shape[0]
    chunks: list[np.ndarray] = []
    for start in range(0, t, batch_size):
        end = min(start + batch_size, t)
        b = end - start
        betas_batch = (
            torch.tensor(betas10[:10], dtype=torch.float32, device=device)
            .unsqueeze(0).expand(b, -1)
        )
        poses_batch = torch.tensor(poses_smpl72[start:end], dtype=torch.float32, device=device)
        trans_batch = torch.tensor(trans[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = body_model(
                global_orient=poses_batch[:, :3],
                body_pose=poses_batch[:, 3:72],
                betas=betas_batch,
                transl=trans_batch,
            )
        chunks.append(out.joints[:, :24].cpu().numpy())
    return np.concatenate(chunks, axis=0)


def joints_to_markers(
    joints: np.ndarray,
    layout: tuple[MarkerSpec, ...],
) -> np.ndarray:
    """Place virtual markers at ``joints[anchor] + offset`` per :class:`MarkerSpec`.

    World-frame offset only — the v1 approximation. Real markers attach to a
    fixed point on the skin (i.e. a SMPL mesh vertex) and rotate with the
    parent bone; that path is deferred until vertex IDs are populated in
    :mod:`csd2smpl.data.marker_layouts`.

    Parameters
    ----------
    joints : np.ndarray
        Shape ``(T, 24, 3)``, SMPL joint world positions.
    layout : tuple of MarkerSpec
        Marker layout to materialise.

    Returns
    -------
    np.ndarray
        Shape ``(T, M, 3)``, dtype float32.
    """
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"expected joints shape (T, J, 3); got {joints.shape}")
    anchors, offsets, _ = layout_to_arrays(layout)
    return (joints[:, anchors, :] + offsets[None, :, :]).astype(np.float32)


def add_noise(
    markers: np.ndarray,
    noise_std: float = 0.01,
    dropout_p: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian per-marker noise + per-frame per-marker dropout (occlusion sim).

    Parameters
    ----------
    markers : np.ndarray
        Shape ``(T, M, 3)``.
    noise_std : float
        Gaussian std in metres (0.01 ≈ 10 mm — realistic for optical mocap).
    dropout_p : float
        Per-frame per-marker probability of being zeroed (occlusion).
    rng : np.random.Generator or None
        Optional RNG; ``None`` uses a default-seeded generator.

    Returns
    -------
    noisy : np.ndarray
        Shape ``(T, M, 3)``, float32. Dropped markers are zeroed.
    mask : np.ndarray
        Shape ``(T, M)``, bool. ``True`` = marker valid.
    """
    if rng is None:
        rng = np.random.default_rng()
    noisy = markers + rng.normal(0, noise_std, markers.shape).astype(np.float32)
    mask = rng.random(markers.shape[:2]) > dropout_p
    noisy[~mask] = 0.0
    return noisy.astype(np.float32), mask.astype(bool)


def synthesize_file(
    npz_path: Path,
    out_path: Path,
    model_path: str,
    layout_name: str = "cmu_41",
    noise_std: float = 0.01,
    dropout_p: float = 0.05,
    target_fps: float = 30.0,
    device: str = "cpu",
    min_frames: int = 8,
    forward_fn: Callable[..., np.ndarray] | None = None,
) -> bool:
    """Convert one AMASS NPZ to a markers+SMPL-targets NPZ.

    Parameters
    ----------
    npz_path : Path
        Source AMASS NPZ.
    out_path : Path
        Destination NPZ path; parent directories are created.
    model_path : str
        SMPL body-model directory passed to smplx.
    layout_name : str
        Key into :data:`csd2smpl.data.marker_layouts.LAYOUTS`. Default ``"cmu_41"``.
    noise_std, dropout_p : float
        Forwarded to :func:`add_noise`.
    target_fps : float
        Downsample to this rate. ``<=0`` disables downsampling.
    device : str
        Torch device for the smplx forward pass.
    min_frames : int
        Sequences shorter than this (after downsampling) are skipped.
    forward_fn : callable, optional
        Override for the joint forward-kinematics function. Defaults to
        :func:`smpl_forward_joints`. Tests inject a stub to bypass smplx.

    Returns
    -------
    bool
        ``True`` if the file was written, ``False`` if skipped.
    """
    layout = get_layout(layout_name)

    seq = np.load(npz_path, allow_pickle=True)
    poses_amass = np.asarray(seq["poses"]).astype(np.float32)
    betas = np.asarray(seq["betas"]).astype(np.float32)
    trans = np.asarray(seq["trans"]).astype(np.float32)
    gender_raw = seq["gender"] if "gender" in seq.files else "neutral"
    gender = gender_raw.item() if hasattr(gender_raw, "item") else gender_raw
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")

    src_fps = get_amass_framerate(seq)
    poses_smpl72 = amass_to_smpl72(poses_amass)
    poses_smpl72 = downsample(poses_smpl72, src_fps, target_fps)
    trans = downsample(trans, src_fps, target_fps)
    if poses_smpl72.shape[0] < min_frames:
        return False

    fwd = forward_fn or smpl_forward_joints
    joints = fwd(
        poses_smpl72=poses_smpl72,
        betas10=betas[:10],
        trans=trans,
        model_path=model_path,
        gender=gender,
        device=device,
    )

    markers_clean = joints_to_markers(joints, layout)
    markers_noisy, mask = add_noise(
        markers_clean, noise_std=noise_std, dropout_p=dropout_p
    )

    _, _, names = layout_to_arrays(layout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        markers=markers_noisy,                  # (T, M, 3)
        markers_clean=markers_clean,            # (T, M, 3)
        mask=mask,                              # (T, M)
        poses=poses_smpl72.astype(np.float32),  # (T, 72)
        betas=betas[:10].astype(np.float32),    # (10,)
        trans=trans.astype(np.float32),         # (T, 3)
        layout=str(layout_name),
        marker_names=",".join(names),
        gender=str(gender),
        fps=np.float32(target_fps if target_fps > 0 else src_fps),
    )
    return True
