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
    layout_vertex_ids,
    load_cmu41_with_vertex_ids,
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



def smpl_forward(
    poses_smpl72: np.ndarray,
    betas10: np.ndarray,
    trans: np.ndarray,
    model_path: str,
    gender: str = "neutral",
    batch_size: int = 256,
    device: str = "cpu",
    return_vertices: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run SMPL forward kinematics; return joints and optionally vertices.

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
    return_vertices : bool
        If ``True``, also return the 6890-vertex mesh per frame.

    Returns
    -------
    joints : np.ndarray
        Shape ``(T, 24, 3)``, metres.
    vertices : np.ndarray or None
        Shape ``(T, 6890, 3)`` if ``return_vertices``, else ``None``.
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
    j_chunks: list[np.ndarray] = []
    v_chunks: list[np.ndarray] | None = [] if return_vertices else None
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
        j_chunks.append(out.joints[:, :24].cpu().numpy())
        if v_chunks is not None:
            v_chunks.append(out.vertices.cpu().numpy())

    joints = np.concatenate(j_chunks, axis=0)
    vertices = np.concatenate(v_chunks, axis=0) if v_chunks is not None else None
    return joints, vertices


def smpl_forward_joints(
    poses_smpl72: np.ndarray,
    betas10: np.ndarray,
    trans: np.ndarray,
    model_path: str,
    gender: str = "neutral",
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Back-compat shim that returns just joints. Prefer :func:`smpl_forward`."""
    joints, _ = smpl_forward(
        poses_smpl72=poses_smpl72, betas10=betas10, trans=trans,
        model_path=model_path, gender=gender,
        batch_size=batch_size, device=device, return_vertices=False,
    )
    return joints


def joints_to_markers(
    joints: np.ndarray,
    layout: tuple[MarkerSpec, ...],
) -> np.ndarray:
    """Place virtual markers at ``joints[anchor] + offset`` per :class:`MarkerSpec`.

    World-frame offset only — the v1 approximation. Use
    :func:`vertices_to_markers` when ``layout`` has vertex IDs populated
    (i.e. after ``fetch_external.sh`` + :func:`load_cmu41_with_vertex_ids`).

    Parameters
    ----------
    joints : np.ndarray
        Shape ``(T, 24, 3)``.
    layout : tuple of MarkerSpec
        Marker layout.

    Returns
    -------
    np.ndarray
        Shape ``(T, M, 3)``, dtype float32.
    """
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"expected joints shape (T, J, 3); got {joints.shape}")
    anchors, offsets, _ = layout_to_arrays(layout)
    return (joints[:, anchors, :] + offsets[None, :, :]).astype(np.float32)


def vertices_to_markers(
    vertices: np.ndarray,
    joints: np.ndarray,
    layout: tuple[MarkerSpec, ...],
) -> np.ndarray:
    """Place markers at SMPL mesh vertex IDs; fall back to joint+offset when None.

    The SSM mapping doesn't cover every CMU label (e.g. RUPA ≠ RUPA2 is a
    close alias; some markers have no SSM equivalent). Markers whose
    ``vertex_id`` is ``None`` use the joint-anchored fallback so every
    position in the returned ``(T, M, 3)`` tensor is defined.

    Parameters
    ----------
    vertices : np.ndarray
        Shape ``(T, 6890, 3)`` — SMPL mesh vertex positions per frame.
    joints : np.ndarray
        Shape ``(T, 24, 3)`` — used only for markers lacking a vertex ID.
    layout : tuple of MarkerSpec
        Marker layout; entries with populated ``vertex_id`` use the mesh,
        others fall back to joint-anchored offsets.

    Returns
    -------
    np.ndarray
        Shape ``(T, M, 3)``, dtype float32.
    """
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"expected vertices shape (T, V, 3); got {vertices.shape}")
    if vertices.shape[0] != joints.shape[0]:
        raise ValueError(
            f"frame count mismatch: vertices T={vertices.shape[0]} "
            f"vs joints T={joints.shape[0]}"
        )
    t = vertices.shape[0]
    m = len(layout)
    out = np.empty((t, m, 3), dtype=np.float32)
    for i, spec in enumerate(layout):
        if spec.vertex_id is not None:
            out[:, i, :] = vertices[:, spec.vertex_id, :]
        else:
            offset = np.asarray(spec.offset, dtype=np.float32)
            out[:, i, :] = joints[:, spec.anchor_joint, :] + offset
    return out


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


def resolve_layout(
    layout_name: str,
    ssm_json_path: str | Path | None,
    placement: str,
) -> tuple[tuple[MarkerSpec, ...], str]:
    """Build the final layout + placement mode to use for synthesis.

    Parameters
    ----------
    layout_name : str
        Registered layout key (e.g. ``"cmu_41"``).
    ssm_json_path : str or Path or None
        Path to the SSM vertex-ID JSON. Required when ``placement="vertex"``.
        If ``placement="auto"`` and this is not None + exists, vertex mode
        is used; otherwise joint-anchored.
    placement : str
        ``"vertex"`` (error if JSON missing), ``"joint_offset"`` (never use
        vertex IDs), or ``"auto"`` (vertex if JSON present, else joint).

    Returns
    -------
    layout : tuple of MarkerSpec
        The resolved layout (vertex IDs populated iff vertex mode selected).
    mode : str
        ``"vertex"`` or ``"joint_offset"``.
    """
    if placement not in {"vertex", "joint_offset", "auto"}:
        raise ValueError(
            f"placement must be 'vertex'|'joint_offset'|'auto'; got {placement!r}"
        )
    base = get_layout(layout_name)
    ssm_present = ssm_json_path is not None and Path(ssm_json_path).exists()

    if placement == "joint_offset":
        return base, "joint_offset"
    if placement == "vertex":
        if not ssm_present:
            raise FileNotFoundError(
                f"placement=vertex but SSM JSON not at {ssm_json_path}. "
                f"Run: bash csd2smpl/scripts/fetch_external.sh"
            )
        if layout_name != "cmu_41":
            raise NotImplementedError(
                f"vertex-mode is only wired for cmu_41; got layout {layout_name!r}"
            )
        return load_cmu41_with_vertex_ids(ssm_json_path), "vertex"
    # auto
    if ssm_present and layout_name == "cmu_41":
        return load_cmu41_with_vertex_ids(ssm_json_path), "vertex"
    return base, "joint_offset"


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
    placement: str = "auto",
    ssm_json_path: str | Path | None = None,
    forward_fn: Callable[..., tuple[np.ndarray, np.ndarray | None]] | None = None,
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
        Key into :data:`csd2smpl.data.marker_layouts.LAYOUTS`.
    noise_std, dropout_p : float
        Forwarded to :func:`add_noise`.
    target_fps : float
        Downsample to this rate. ``<=0`` disables downsampling.
    device : str
        Torch device for the smplx forward pass.
    min_frames : int
        Sequences shorter than this (after downsampling) are skipped.
    placement : str
        ``"vertex"`` | ``"joint_offset"`` | ``"auto"``. See
        :func:`resolve_layout`.
    ssm_json_path : str or Path or None
        Path to SSM vertex-ID JSON (required for ``placement='vertex'``).
    forward_fn : callable, optional
        Override for the SMPL forward. Signature must match
        :func:`smpl_forward`: returns ``(joints, vertices_or_None)``. Tests
        inject a stub to bypass smplx.

    Returns
    -------
    bool
        ``True`` if the file was written, ``False`` if skipped.
    """
    layout, mode = resolve_layout(layout_name, ssm_json_path, placement)

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

    need_vertices = (mode == "vertex")
    fwd = forward_fn or smpl_forward
    joints, vertices = fwd(
        poses_smpl72=poses_smpl72,
        betas10=betas[:10],
        trans=trans,
        model_path=model_path,
        gender=gender,
        device=device,
        return_vertices=need_vertices,
    )

    if mode == "vertex":
        if vertices is None:
            raise RuntimeError("forward_fn did not return vertices despite return_vertices=True")
        markers_clean = vertices_to_markers(vertices, joints, layout)
    else:
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
        placement=str(mode),
        marker_names=",".join(names),
        gender=str(gender),
        fps=np.float32(target_fps if target_fps > 0 else src_fps),
    )
    return True
