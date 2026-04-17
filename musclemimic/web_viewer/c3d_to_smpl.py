"""Convert C3D marker data to SMPL parameters with a MoSh++-style pipeline.

This module does not embed the original chumpy-based `moshpp` implementation.
Instead, it mirrors the parts that are practical in this repo's existing torch
stack:

1. Canonical marker labels and fixed SMPL/SMPLH marker vertices.
2. Marker-type-dependent distance-to-skin handling.
3. Stage-I frame picking based on marker availability.
4. Stage-I optimization over shared betas, per-frame rigid pose, and latent
   marker coefficients anchored to the mesh surface.
5. Stage-II warm-started per-frame pose fitting with missing-marker-dependent
   weight annealing and temporal smoothing.

Outputs an AMASS-compatible dict that feeds directly into the existing
retargeting pipeline (fit_smpl_motion / fit_gmr_motion).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as sRot

logger = logging.getLogger(__name__)

try:
    import ezc3d
except ImportError:  # pragma: no cover - exercised in environments without the c3d extra.
    ezc3d = None


# Marker-type-dependent defaults copied from the public moshpp config.
MOSHPP_STAGEI_WT_ANNEALING = (1.0, 0.5, 0.25, 0.125)
MOSHPP_STAGEI_WT_POSE_BODY = 3.0
MOSHPP_STAGEI_WT_BETAS = 10.0
MOSHPP_STAGEI_WT_INIT = 300.0
MOSHPP_STAGEI_WT_DATA = 75.0
MOSHPP_STAGEI_WT_SURF = 10000.0
MOSHPP_STAGEII_WT_DATA = 400.0
MOSHPP_STAGEII_WT_POSE_BODY = 1.6
MOSHPP_STAGEII_WT_VELO = 2.5
MOSHPP_STAGEII_WT_ANNEALING = 2.5
MOSHPP_NUM_TRAIN_MARKERS = 46

MARKER_TYPE_DISTANCES = {
    "body": 0.0095,
    "wrist": 0.0390,
}

WRIST_MARKER_LABELS = {"LIWR", "LOWR", "RIWR", "ROWR"}

# Subset of moshpp marker-label aliases that matter for standard Plug-in Gait /
# gait-lab body marker sets.
MOSHPP_LABEL_MAP: dict[str, str] = {
    "HEAD_TOP": "ARIEL",
    "TOPBACK": "C7",
    "NECK_BASE": "C7",
    "CHEST": "CLAV",
    "UPTHRX": "CLAV",
    "STERNUM": "STRN",
    "SETRNUM": "STRN",
    "LOTHRX": "STRN",
    "LOBACK": "T10",
    "MIDBACK": "T8",
    "MID_BACK": "T8",
    "LASI": "LFWT",
    "LPSI": "LBWT",
    "RASI": "RFWT",
    "RPSI": "RBWT",
    "LWRA": "LIWR",
    "LWRB": "LOWR",
    "RWRA": "RIWR",
    "RWRB": "ROWR",
    "LMELB": "LELBIN",
    "RMELB": "RELBIN",
    "LMKNE": "LKNI",
    "RMKNE": "RKNI",
    "LMANK": "LHEEI",
    "RMANK": "RHEEI",
    "LANKIN": "LHEEI",
    "RANKIN": "RHEEI",
    "L_INWRIST": "LIWR",
    "R_INWRIST": "RIWR",
    "L_OUTWRIST": "LOWR",
    "R_OUTWRIST": "ROWR",
    "L_INKNEE": "LKNI",
    "R_INKNEE": "RKNI",
    "L_OUTKNEE": "LKNE",
    "R_OUTKNEE": "RKNE",
    "L_ANKLE": "LANK",
    "R_ANKLE": "RANK",
    "L_KNEE": "LKNE",
    "R_KNEE": "RKNE",
    "L_HEEL": "LHEE",
    "R_HEEL": "RHEE",
    "L_TOETIP": "LTOE",
    "R_TOETIP": "RTOE",
}

# SMPL / SMPLH marker vertices used by moshpp after label canonicalization.
MOSHPP_SMPLH_MARKER_VIDS: dict[str, int] = {
    "C7": 3470,
    "CLAV": 3171,
    "LBHD": 182,
    "LFHD": 0,
    "RBHD": 3694,
    "RFHD": 3512,
    "STRN": 3506,
    "T10": 3016,
    "RBAK": 5273,
    "LBWT": 3122,
    "LFWT": 857,
    "RBWT": 6544,
    "RFWT": 4343,
    "LSHO": 1861,
    "RSHO": 5322,
    "LUPA": 1443,
    "RUPA": 4918,
    "LELB": 1666,
    "LELBIN": 1725,
    "RELB": 5135,
    "RELBIN": 5194,
    "LFRM": 1568,
    "RFRM": 5037,
    "LIWR": 2112,
    "LOWR": 2108,
    "RIWR": 5573,
    "ROWR": 5568,
    "LFIN": 2174,
    "RFIN": 5635,
    "LTHI": 1454,
    "RTHI": 4927,
    "LKNE": 1053,
    "LKNI": 1058,
    "RKNE": 4538,
    "RKNI": 4544,
    "LTIB": 1112,
    "RTIB": 4598,
    "LANK": 3327,
    "RANK": 6728,
    "LHEEI": 3432,
    "RHEEI": 6832,
    "LHEE": 3387,
    "RHEE": 6786,
    "LTOE": 3233,
    "RTOE": 6633,
}


@dataclass(frozen=True)
class MarkerLayout:
    marker_vids: dict[str, int]
    marker_type: dict[str, str]
    marker_type_mask: dict[str, np.ndarray]
    m2b_distance: dict[str, float]
    surface_model_type: str = "smplh"

    @property
    def labels(self) -> list[str]:
        return list(self.marker_vids.keys())


@dataclass(frozen=True)
class StageIWeights:
    anneal_factor: float
    data: float
    pose_body: float
    betas: float
    init_by_type: dict[str, float]
    surf: float


@dataclass(frozen=True)
class StageIIWeights:
    anneal_factor: float
    data: float
    pose_body: float
    velo: float


@dataclass(frozen=True)
class PreparedMarkerObservations:
    positions: np.ndarray
    labels: list[str]
    unknown_labels: list[str]


@dataclass(frozen=True)
class SurfaceMarkerModel:
    vids: np.ndarray
    neighbor_vids: np.ndarray
    reference_normals: np.ndarray
    initial_coeffs: np.ndarray
    desired_distances: np.ndarray
    marker_types: tuple[str, ...]

    @staticmethod
    def from_layout(
        smpl_model,
        marker_layout: MarkerLayout,
        betas=None,
        pose_dim: int = 156,
        device: str = "cpu",
    ) -> SurfaceMarkerModel:
        import torch

        if betas is None:
            betas = torch.zeros(1, 16, device=device, dtype=torch.float32)

        with torch.no_grad():
            verts, _ = smpl_model.get_joints_verts(
                torch.zeros(1, pose_dim, device=device, dtype=torch.float32),
                th_betas=betas,
            )
            verts_np = verts[0].detach().cpu().numpy()

        faces = smpl_model.faces.astype(np.int64)
        vert_normals = _compute_vertex_normals_from_mesh(faces, verts_np)
        vertex_neighbors = _compute_vertex_neighbors(faces, len(verts_np))
        tree = KDTree(verts_np)

        labels = marker_layout.labels
        vids = np.asarray([marker_layout.marker_vids[label] for label in labels], dtype=np.int64)
        neighbor_vids = np.zeros((len(labels), 2), dtype=np.int64)
        reference_normals = np.zeros((len(labels), 3), dtype=np.float32)
        initial_coeffs = np.zeros((len(labels), 3), dtype=np.float32)
        desired_distances = np.zeros(len(labels), dtype=np.float32)
        marker_types: list[str] = []

        for idx, label in enumerate(labels):
            vid = vids[idx]
            neighbors = _pick_frame_neighbors(vid, vertex_neighbors, verts_np, tree)
            neighbor_vids[idx] = neighbors

            v0 = verts_np[vid]
            v1 = verts_np[neighbors[0]]
            v2 = verts_np[neighbors[1]]
            _, _, n = _build_local_frame_np(v0, v1, v2)
            if np.dot(n, vert_normals[vid]) < 0:
                n = -n

            marker_type = marker_layout.marker_type[label]
            desired_distances[idx] = marker_layout.m2b_distance[marker_type]
            reference_normals[idx] = n
            initial_coeffs[idx] = np.array([0.0, 0.0, desired_distances[idx]], dtype=np.float32)
            marker_types.append(marker_type)

        return SurfaceMarkerModel(
            vids=vids,
            neighbor_vids=neighbor_vids,
            reference_normals=reference_normals,
            initial_coeffs=initial_coeffs,
            desired_distances=desired_distances,
            marker_types=tuple(marker_types),
        )

    def reconstruct(self, posed_verts, coeffs=None):
        import torch

        squeeze = posed_verts.dim() == 2
        if squeeze:
            posed_verts = posed_verts.unsqueeze(0)

        coeffs_t = (
            torch.as_tensor(self.initial_coeffs, dtype=torch.float32, device=posed_verts.device)
            if coeffs is None
            else coeffs.to(posed_verts.device)
        )
        if coeffs_t.dim() == 2:
            coeffs_t = coeffs_t.unsqueeze(0).expand(posed_verts.shape[0], -1, -1)

        vids_t = torch.as_tensor(self.vids, dtype=torch.long, device=posed_verts.device)
        nn_t = torch.as_tensor(self.neighbor_vids, dtype=torch.long, device=posed_verts.device)
        ref_n = torch.as_tensor(self.reference_normals, dtype=torch.float32, device=posed_verts.device)

        v0 = posed_verts[:, vids_t]
        v1 = posed_verts[:, nn_t[:, 0]]
        v2 = posed_verts[:, nn_t[:, 1]]

        t1, t2, n = _build_local_frame_torch(v0, v1, v2, ref_n)
        markers = v0 + coeffs_t[:, :, 0:1] * t1 + coeffs_t[:, :, 1:2] * t2 + coeffs_t[:, :, 2:3] * n
        return markers.squeeze(0) if squeeze else markers


def canonicalize_marker_label(label: str, labels_map: dict[str, str] | None = None) -> str:
    normalized = label.replace(" ", "")
    normalized = normalized.split(":")[-1]
    normalized = normalized.upper()
    if labels_map is None:
        labels_map = MOSHPP_LABEL_MAP
    return labels_map.get(normalized, normalized)


def compute_marker_availability_mask(markers: np.ndarray) -> np.ndarray:
    """True where a marker is available in a frame."""
    markers = np.asarray(markers, dtype=np.float32)
    return ~(np.isnan(markers).any(axis=-1) | np.all(np.isclose(markers, 0.0), axis=-1))


def compute_stagei_weights(
    num_markers: int,
    marker_types: Iterable[str],
    anneal_factor: float,
) -> StageIWeights:
    marker_types = tuple(dict.fromkeys(marker_types))
    init_by_type = dict.fromkeys(marker_types, MOSHPP_STAGEI_WT_INIT * anneal_factor)
    return StageIWeights(
        anneal_factor=anneal_factor,
        data=(MOSHPP_STAGEI_WT_DATA / anneal_factor) * (MOSHPP_NUM_TRAIN_MARKERS / max(1, num_markers)),
        pose_body=MOSHPP_STAGEI_WT_POSE_BODY * anneal_factor,
        betas=MOSHPP_STAGEI_WT_BETAS * anneal_factor,
        init_by_type=init_by_type,
        surf=MOSHPP_STAGEI_WT_SURF,
    )


def compute_stageii_weights(num_observed_markers: int, num_total_markers: int) -> StageIIWeights:
    missing_ratio = 0.0 if num_total_markers == 0 else (num_total_markers - num_observed_markers) / num_total_markers
    anneal_factor = 1.0 + missing_ratio * MOSHPP_STAGEII_WT_ANNEALING
    return StageIIWeights(
        anneal_factor=anneal_factor,
        data=MOSHPP_STAGEII_WT_DATA * (MOSHPP_NUM_TRAIN_MARKERS / max(1, num_observed_markers)),
        pose_body=MOSHPP_STAGEII_WT_POSE_BODY * anneal_factor,
        velo=MOSHPP_STAGEII_WT_VELO,
    )


def build_marker_layout(
    labels: Iterable[str],
    *,
    labels_map: dict[str, str] | None = None,
    wrist_markers_on_stick: bool = False,
) -> MarkerLayout:
    canonical_labels = [canonicalize_marker_label(label, labels_map=labels_map) for label in labels]
    available_labels = sorted({label for label in canonical_labels if label in MOSHPP_SMPLH_MARKER_VIDS})

    marker_type = {}
    for label in available_labels:
        marker_type[label] = "wrist" if wrist_markers_on_stick and label in WRIST_MARKER_LABELS else "body"

    marker_vids = {label: MOSHPP_SMPLH_MARKER_VIDS[label] for label in available_labels}
    marker_type_mask = {}
    for marker_kind in sorted(set(marker_type.values())):
        marker_type_mask[marker_kind] = np.array(
            [marker_type[label] == marker_kind for label in available_labels],
            dtype=bool,
        )

    return MarkerLayout(
        marker_vids=marker_vids,
        marker_type=marker_type,
        marker_type_mask=marker_type_mask,
        m2b_distance={marker_kind: MARKER_TYPE_DISTANCES[marker_kind] for marker_kind in marker_type_mask},
    )


def prepare_marker_observations(
    positions: np.ndarray,
    labels: Iterable[str],
    *,
    labels_map: dict[str, str] | None = None,
) -> PreparedMarkerObservations:
    groups: dict[str, list[int]] = {}
    unknown_labels: list[str] = []

    for idx, label in enumerate(labels):
        canonical = canonicalize_marker_label(label, labels_map=labels_map)
        if canonical == "NAN":
            continue
        if canonical not in MOSHPP_SMPLH_MARKER_VIDS:
            unknown_labels.append(canonical)
            continue
        groups.setdefault(canonical, []).append(idx)

    ordered_labels = sorted(groups)
    collapsed = np.zeros((positions.shape[0], len(ordered_labels), 3), dtype=np.float32)
    for out_idx, canonical in enumerate(ordered_labels):
        collapsed[:, out_idx] = _nanmean_with_all_nan(positions[:, groups[canonical], :], axis=1)

    return PreparedMarkerObservations(
        positions=collapsed,
        labels=ordered_labels,
        unknown_labels=sorted(set(unknown_labels)),
    )


def pick_stagei_frames(
    markers: np.ndarray,
    *,
    num_frames: int,
    seed: int = 100,
    least_avail_markers: float = 1.0,
    strict: bool = True,
) -> np.ndarray:
    if markers.ndim != 3:
        raise ValueError(f"Expected markers with shape (T, N, 3), got {markers.shape}")

    availability = compute_marker_availability_mask(markers).sum(axis=-1) / max(1, markers.shape[1])
    threshold = float(least_avail_markers)

    eligible = np.flatnonzero(availability >= threshold)
    if strict and len(eligible) < num_frames:
        raise ValueError(
            f"Not enough frames have at least {threshold * 100:.1f}% of the markers: "
            f"requested {num_frames}, found {len(eligible)}"
        )

    while not strict and len(eligible) < num_frames:
        threshold -= 0.01
        if threshold < 0.01:
            raise ValueError("Unable to find enough stage-I frames with sufficient marker coverage.")
        eligible = np.flatnonzero(availability >= threshold)

    rng = np.random.default_rng(seed)
    picks = rng.choice(eligible, size=min(num_frames, len(eligible)), replace=False)
    return np.sort(picks.astype(np.int64))


def load_c3d_markers(path: str) -> tuple[np.ndarray, list[str], float]:
    """Load C3D markers as (frames, markers, xyz) in meters, preserving Z-up world axes."""
    if ezc3d is None:
        raise ImportError("ezc3d is required for C3D support. Install the optional `c3d` extra.")

    c3d = ezc3d.c3d(path)
    labels = [str(label) for label in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    fps = float(c3d["header"]["points"]["frame_rate"])

    points = np.asarray(c3d["data"]["points"][:3], dtype=np.float32).transpose(2, 1, 0)
    units = c3d["parameters"]["POINT"].get("UNITS", {}).get("value", ["mm"])
    unit = str(units[0]).lower() if units else "mm"
    scale = {"mm": 1000.0, "cm": 100.0, "m": 1.0}.get(unit, 1000.0)
    points /= scale

    residuals = c3d["data"].get("meta_points", {}).get("residuals")
    if residuals is not None:
        residuals = np.asarray(residuals, dtype=np.float32).transpose(2, 1, 0)[..., 0]
        points[residuals < 0] = np.nan

    zero_mask = np.all(np.isclose(points, 0.0), axis=-1)
    points[zero_mask] = np.nan

    return points, labels, fps


def fit_smpl_to_c3d(
    c3d_path: str,
    smpl_model_path: str,
    stage1_iters: int = 320,
    stage2_iters: int = 80,
    stage1_lr: float = 0.01,
    stage2_lr: float = 0.01,
    n_ref_frames: int = 12,
    target_fps: float = 30.0,
    device: str = "cpu",
    seed: int = 100,
    least_avail_markers: float = 1.0,
    wrist_markers_on_stick: bool = False,
) -> dict:
    import torch

    from loco_mujoco.smpl import SMPLH_Parser

    pose_dim = 156
    body_end = 66

    raw_positions, raw_labels, fps = load_c3d_markers(c3d_path)
    prepared = prepare_marker_observations(raw_positions, raw_labels)

    # Downsample to target_fps if capture rate is much higher (e.g. 360 Hz → 30 Hz)
    if fps > target_fps * 1.5:
        skip = max(1, round(fps / target_fps))
        prepared = PreparedMarkerObservations(
            positions=prepared.positions[::skip],
            labels=prepared.labels,
            unknown_labels=prepared.unknown_labels,
        )
        effective_fps = fps / skip
        logger.info("Downsampled %.0f Hz → %.0f Hz (skip=%d, %d → %d frames)", fps, effective_fps, skip, raw_positions.shape[0], prepared.positions.shape[0])
    else:
        effective_fps = fps
    if prepared.positions.shape[1] < 3:
        raise ValueError(
            f"Need at least 3 usable markers after canonicalization, got {prepared.positions.shape[1]} "
            f"from labels {prepared.labels}"
        )

    marker_layout = build_marker_layout(prepared.labels, wrist_markers_on_stick=wrist_markers_on_stick)
    label_to_index = {label: idx for idx, label in enumerate(prepared.labels)}
    ordered_labels = marker_layout.labels
    observed = np.stack([prepared.positions[:, label_to_index[label]] for label in ordered_labels], axis=1).astype(np.float32)
    availability = compute_marker_availability_mask(observed)

    logger.info(
        "Loaded %s: %d frames, %d/%d markers matched to the moshpp layout @ %.1f Hz",
        c3d_path,
        observed.shape[0],
        len(ordered_labels),
        len(raw_labels),
        fps,
    )
    logger.info(
        "Planned optimization work: Stage I %d ref frames x %d iters, Stage II %d frames x %d iters/frame",
        min(n_ref_frames, observed.shape[0]),
        stage1_iters,
        observed.shape[0],
        stage2_iters,
    )
    if prepared.unknown_labels:
        logger.info("Ignored unknown markers after canonicalization: %s", ", ".join(prepared.unknown_labels))

    ref_indices = pick_stagei_frames(
        observed,
        num_frames=min(n_ref_frames, observed.shape[0]),
        seed=seed,
        least_avail_markers=least_avail_markers,
        strict=True,
    )

    smpl = SMPLH_Parser(model_path=smpl_model_path, gender="neutral", create_transl=False).to(device)
    marker_model = SurfaceMarkerModel.from_layout(smpl, marker_layout, pose_dim=pose_dim, device=device)

    observed_t = torch.as_tensor(observed, dtype=torch.float32, device=device)
    betas_opt, coeffs_opt, stage1_debug = _fit_stagei(
        smpl=smpl,
        marker_model=marker_model,
        observed_t=observed_t,
        availability=availability,
        ref_indices=ref_indices,
        pose_dim=pose_dim,
        body_end=body_end,
        stage1_iters=stage1_iters,
        stage1_lr=stage1_lr,
        device=device,
    )

    pose_aa_156, trans, stage2_debug = _fit_stageii(
        smpl=smpl,
        marker_model=marker_model,
        observed_t=observed_t,
        availability=availability,
        betas_opt=betas_opt,
        coeffs_opt=coeffs_opt,
        pose_dim=pose_dim,
        body_end=body_end,
        stage2_iters=stage2_iters,
        stage2_lr=stage2_lr,
        device=device,
    )

    pose_aa = np.concatenate([pose_aa_156[:, :body_end], np.zeros((pose_aa_156.shape[0], 6), dtype=np.float32)], axis=-1)
    return {
        "pose_aa": pose_aa,
        "trans": trans,
        "betas": betas_opt.detach().cpu().numpy().reshape(-1),
        "fps": float(effective_fps),
        "gender": "neutral",
        "debug": {
            "canonical_labels": ordered_labels,
            "stagei_ref_indices": ref_indices,
            "stagei": stage1_debug,
            "stageii": stage2_debug,
        },
    }


def fit_smpl_to_c3d_cached(
    c3d_path: str,
    smpl_model_path: str,
    cache_dir: str | None = None,
    **kwargs,
) -> dict:
    """Fit SMPL to C3D with npz caching."""
    c3d_name = Path(c3d_path).stem
    if cache_dir is None:
        cache_dir = str(Path(c3d_path).parent / ".smpl_cache")

    cache_path = Path(cache_dir) / f"{c3d_name}_smpl.npz"
    if cache_path.exists():
        logger.info("Loading cached SMPL fit from %s", cache_path)
        data = np.load(cache_path, allow_pickle=True)
        return {
            "pose_aa": data["pose_aa"],
            "trans": data["trans"],
            "betas": data["betas"],
            "fps": float(data["fps"]),
            "gender": str(data["gender"]),
        }

    result = fit_smpl_to_c3d(c3d_path, smpl_model_path, **kwargs)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        pose_aa=result["pose_aa"],
        trans=result["trans"],
        betas=result["betas"],
        fps=result["fps"],
        gender=result["gender"],
    )
    logger.info("Cached SMPL fit to %s", cache_path)
    return result


def save_motion_data_as_amass_smplh_npz(
    motion_data: dict,
    output_path: str | Path,
) -> Path:
    """Write fitted motion data to an AMASS/GMR-compatible SMPL-H npz file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pose_aa = np.asarray(motion_data["pose_aa"], dtype=np.float32)
    if pose_aa.ndim != 2:
        raise ValueError(f"Expected pose_aa with shape (T, D), got {pose_aa.shape}")

    if pose_aa.shape[1] == 156:
        poses = pose_aa
    elif pose_aa.shape[1] <= 156:
        poses = np.concatenate(
            [pose_aa, np.zeros((pose_aa.shape[0], 156 - pose_aa.shape[1]), dtype=np.float32)],
            axis=-1,
        )
    else:
        raise ValueError(f"pose_aa has too many columns for SMPL-H export: {pose_aa.shape[1]}")

    betas = np.asarray(motion_data["betas"], dtype=np.float32)
    trans = np.asarray(motion_data["trans"], dtype=np.float32)
    fps = float(motion_data["fps"])
    gender = str(motion_data.get("gender", "neutral"))

    np.savez(
        output_path,
        poses=poses,
        trans=trans,
        betas=betas,
        gender=np.array(gender),
        mocap_framerate=np.array(fps, dtype=np.float32),
    )
    return output_path


def _fit_stagei(
    *,
    smpl,
    marker_model: SurfaceMarkerModel,
    observed_t,
    availability: np.ndarray,
    ref_indices: np.ndarray,
    pose_dim: int,
    body_end: int,
    stage1_iters: int,
    stage1_lr: float,
    device: str,
):
    import torch

    logger.info("Stage I: fitting shape across %d reference frames (%d iters)...", len(ref_indices), stage1_iters)
    ref_observed = observed_t[ref_indices]
    ref_availability = torch.as_tensor(availability[ref_indices], dtype=torch.bool, device=device)

    betas = torch.zeros(1, 16, dtype=torch.float32, device=device, requires_grad=True)
    coeffs = torch.as_tensor(marker_model.initial_coeffs, dtype=torch.float32, device=device).clone().requires_grad_(True)

    ref_poses = []
    ref_trans = []
    for local_idx, frame_idx in enumerate(ref_indices):
        pose_init, trans_init = _initialize_pose_and_trans(
            smpl=smpl,
            marker_model=marker_model,
            betas=torch.zeros(1, 16, dtype=torch.float32, device=device),
            coeffs=torch.as_tensor(marker_model.initial_coeffs, dtype=torch.float32, device=device),
            observed_frame=ref_observed[local_idx],
            availability_mask=ref_availability[local_idx],
            pose_dim=pose_dim,
            device=device,
        )
        ref_poses.append(pose_init.requires_grad_(True))
        ref_trans.append(trans_init.requires_grad_(True))
        logger.debug("Stage I init frame %d -> global orient norm %.4f", int(frame_idx), pose_init[:, :3].norm().item())

    optimizer = torch.optim.Adam([betas, coeffs, *ref_poses, *ref_trans], lr=stage1_lr)
    coeff_init = torch.as_tensor(marker_model.initial_coeffs, dtype=torch.float32, device=device)
    coeff_surface_target = torch.as_tensor(marker_model.desired_distances, dtype=torch.float32, device=device)

    iters_per_anneal = max(1, stage1_iters // len(MOSHPP_STAGEI_WT_ANNEALING))
    stage1_losses: list[float] = []

    global_iter = 0
    for phase_idx, anneal_factor in enumerate(MOSHPP_STAGEI_WT_ANNEALING):
        weights = compute_stagei_weights(len(marker_model.vids), marker_model.marker_types, anneal_factor)
        coeff_type_weights = torch.as_tensor(
            [weights.init_by_type[marker_type] for marker_type in marker_model.marker_types],
            dtype=torch.float32,
            device=device,
        )
        logger.info("  Stage I phase %d/%d (anneal=%.3f, %d iters)", phase_idx + 1, len(MOSHPP_STAGEI_WT_ANNEALING), anneal_factor, iters_per_anneal)
        for _ in range(iters_per_anneal):
            optimizer.zero_grad()
            total_loss = torch.zeros((), dtype=torch.float32, device=device)
            data_terms = []

            for ref_idx in range(len(ref_indices)):
                verts, _ = smpl.get_joints_verts(ref_poses[ref_idx], th_betas=betas, th_trans=ref_trans[ref_idx])
                pred_markers = marker_model.reconstruct(verts[0], coeffs=coeffs)
                mask = ref_availability[ref_idx]
                if not torch.any(mask):
                    continue
                residual = pred_markers[mask] - ref_observed[ref_idx][mask]
                data_term = residual.pow(2).sum(dim=-1).mean()
                data_terms.append(data_term)

            if not data_terms:
                raise ValueError("No valid Stage-I reference frames contain usable markers.")

            data_loss = torch.stack(data_terms).mean() * weights.data
            pose_loss = torch.stack([pose[:, 3:body_end].pow(2).mean() for pose in ref_poses]).mean() * weights.pose_body
            beta_loss = betas.pow(2).mean() * weights.betas
            init_loss = ((coeffs - coeff_init).pow(2).sum(dim=-1) * coeff_type_weights).mean()
            surf_loss = (coeffs[:, 2] - coeff_surface_target).pow(2).mean() * weights.surf

            total_loss = data_loss + pose_loss + beta_loss + init_loss + surf_loss
            total_loss.backward()
            optimizer.step()
            stage1_losses.append(float(total_loss.detach().cpu().item()))
            global_iter += 1
            if global_iter % 40 == 0:
                logger.info("    iter %d/%d  loss=%.5f  data=%.5f  beta=%.5f", global_iter, stage1_iters, total_loss.item(), data_loss.item(), beta_loss.item())

    debug = {
        "loss_curve": np.asarray(stage1_losses, dtype=np.float32),
        "ref_indices": ref_indices.copy(),
    }
    return betas.detach(), coeffs.detach(), debug


def _fit_stageii(
    *,
    smpl,
    marker_model: SurfaceMarkerModel,
    observed_t,
    availability: np.ndarray,
    betas_opt,
    coeffs_opt,
    pose_dim: int,
    body_end: int,
    stage2_iters: int,
    stage2_lr: float,
    device: str,
):
    import torch

    n_frames = observed_t.shape[0]
    logger.info("Stage II: fitting %d frames (%d iters/frame)...", n_frames, stage2_iters)
    pose_out = np.zeros((n_frames, pose_dim), dtype=np.float32)
    trans_out = np.zeros((n_frames, 3), dtype=np.float32)
    stage2_losses = np.full(n_frames, np.nan, dtype=np.float32)
    missing_markers = np.zeros(n_frames, dtype=np.int64)

    coeffs_opt = coeffs_opt.to(device)
    availability_t = torch.as_tensor(availability, dtype=torch.bool, device=device)

    prev_pose = None
    prev_prev_pose = None
    prev_trans = None

    for frame_idx in range(n_frames):
        obs_frame = observed_t[frame_idx]
        mask = availability_t[frame_idx]
        missing_markers[frame_idx] = int((~mask).sum().item())

        if not torch.any(mask):
            if prev_pose is None or prev_trans is None:
                raise ValueError(f"Frame {frame_idx} has no visible markers and no previous solution to copy.")
            pose_out[frame_idx] = prev_pose.detach().cpu().numpy().reshape(-1)
            trans_out[frame_idx] = prev_trans.detach().cpu().numpy().reshape(-1)
            continue

        if prev_pose is None or prev_trans is None:
            pose, trans = _initialize_pose_and_trans(
                smpl=smpl,
                marker_model=marker_model,
                betas=betas_opt,
                coeffs=coeffs_opt,
                observed_frame=obs_frame,
                availability_mask=mask,
                pose_dim=pose_dim,
                device=device,
            )
            pose, trans, _ = _optimize_frame(
                smpl=smpl,
                marker_model=marker_model,
                betas_opt=betas_opt,
                coeffs_opt=coeffs_opt,
                obs_frame=obs_frame,
                mask=mask,
                init_pose=pose,
                init_trans=trans,
                body_end=body_end,
                stage2_iters=max(10, stage2_iters // 4),
                stage2_lr=stage2_lr,
                weights=compute_stageii_weights(int(mask.sum().item()), len(marker_model.vids)),
                prev_pose=None,
                prev_prev_pose=None,
                body_pose_weight_scale=10.0,
            )
            pose, trans, _ = _optimize_frame(
                smpl=smpl,
                marker_model=marker_model,
                betas_opt=betas_opt,
                coeffs_opt=coeffs_opt,
                obs_frame=obs_frame,
                mask=mask,
                init_pose=pose.detach(),
                init_trans=trans.detach(),
                body_end=body_end,
                stage2_iters=max(10, stage2_iters // 4),
                stage2_lr=stage2_lr,
                weights=compute_stageii_weights(int(mask.sum().item()), len(marker_model.vids)),
                prev_pose=None,
                prev_prev_pose=None,
                body_pose_weight_scale=5.0,
            )
        else:
            pose = prev_pose.detach().clone()
            trans = prev_trans.detach().clone()

        weights = compute_stageii_weights(int(mask.sum().item()), len(marker_model.vids))
        pose, trans, loss = _optimize_frame(
            smpl=smpl,
            marker_model=marker_model,
            betas_opt=betas_opt,
            coeffs_opt=coeffs_opt,
            obs_frame=obs_frame,
            mask=mask,
            init_pose=pose.detach(),
            init_trans=trans.detach(),
            body_end=body_end,
            stage2_iters=stage2_iters,
            stage2_lr=stage2_lr,
            weights=weights,
            prev_pose=prev_pose,
            prev_prev_pose=prev_prev_pose,
            body_pose_weight_scale=1.0,
        )

        pose_out[frame_idx] = pose.detach().cpu().numpy().reshape(-1)
        trans_out[frame_idx] = trans.detach().cpu().numpy().reshape(-1)
        stage2_losses[frame_idx] = loss
        prev_prev_pose = None if prev_pose is None else prev_pose.detach()
        prev_pose = pose.detach()
        prev_trans = trans.detach()

        if (frame_idx + 1) % 50 == 0 or frame_idx == n_frames - 1:
            logger.info("  Frame %d/%d  loss=%.5f  missing=%d", frame_idx + 1, n_frames, loss, int(missing_markers[frame_idx]))

    return pose_out, trans_out, {"frame_loss": stage2_losses, "missing_markers": missing_markers}


def _optimize_frame(
    *,
    smpl,
    marker_model: SurfaceMarkerModel,
    betas_opt,
    coeffs_opt,
    obs_frame,
    mask,
    init_pose,
    init_trans,
    body_end: int,
    stage2_iters: int,
    stage2_lr: float,
    weights: StageIIWeights,
    prev_pose,
    prev_prev_pose,
    body_pose_weight_scale: float,
):
    import torch

    pose = init_pose.clone().detach().requires_grad_(True)
    trans = init_trans.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pose, trans], lr=stage2_lr)

    final_loss = float("nan")
    for _ in range(stage2_iters):
        optimizer.zero_grad()
        verts, _ = smpl.get_joints_verts(pose, th_betas=betas_opt, th_trans=trans)
        pred_markers = marker_model.reconstruct(verts[0], coeffs=coeffs_opt)

        residual = pred_markers[mask] - obs_frame[mask]
        data_loss = residual.pow(2).sum(dim=-1).mean() * weights.data
        pose_loss = pose[:, 3:body_end].pow(2).mean() * weights.pose_body * body_pose_weight_scale

        velo_loss = torch.zeros((), dtype=torch.float32, device=pose.device)
        if prev_pose is not None and prev_prev_pose is not None:
            expected = 2.0 * prev_pose - prev_prev_pose
            velo_loss = (pose[:, :body_end] - expected[:, :body_end]).pow(2).mean() * weights.velo
        elif prev_pose is not None:
            velo_loss = (pose[:, :body_end] - prev_pose[:, :body_end]).pow(2).mean() * (0.25 * weights.velo)

        loss = data_loss + pose_loss + velo_loss
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    return pose.detach(), trans.detach(), final_loss


def _initialize_pose_and_trans(
    *,
    smpl,
    marker_model: SurfaceMarkerModel,
    betas,
    coeffs,
    observed_frame,
    availability_mask,
    pose_dim: int,
    device: str,
):
    import torch

    with torch.no_grad():
        verts, _ = smpl.get_joints_verts(
            torch.zeros(1, pose_dim, dtype=torch.float32, device=device),
            th_betas=betas,
        )
        template_markers = marker_model.reconstruct(verts[0], coeffs=coeffs).detach().cpu().numpy()

    obs_np = observed_frame.detach().cpu().numpy()
    mask_np = availability_mask.detach().cpu().numpy().astype(bool)
    if mask_np.sum() < 3:
        raise ValueError("At least 3 visible markers are required for rigid initialization.")

    rot, trans = _procrustes_align(template_markers[mask_np], obs_np[mask_np])
    pose_init = np.zeros((1, pose_dim), dtype=np.float32)
    pose_init[0, :3] = sRot.from_matrix(rot).as_rotvec().astype(np.float32)
    trans_init = trans.reshape(1, 3).astype(np.float32)

    return (
        torch.as_tensor(pose_init, dtype=torch.float32, device=device),
        torch.as_tensor(trans_init, dtype=torch.float32, device=device),
    )


def _build_local_frame_np(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e1 = v1 - v0
    e2 = v2 - v0
    t1 = e1 / (np.linalg.norm(e1) + 1e-8)
    n = np.cross(e1, e2)
    n = n / (np.linalg.norm(n) + 1e-8)
    t2 = np.cross(n, t1)
    t2 = t2 / (np.linalg.norm(t2) + 1e-8)
    return t1.astype(np.float32), t2.astype(np.float32), n.astype(np.float32)


def _build_local_frame_torch(v0, v1, v2, ref_n):
    import torch

    e1 = v1 - v0
    e2 = v2 - v0
    t1 = e1 / (e1.norm(dim=-1, keepdim=True) + 1e-8)
    n = torch.cross(e1, e2, dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
    flip_mask = (n * ref_n.unsqueeze(0)).sum(dim=-1, keepdim=True) < 0
    n = torch.where(flip_mask, -n, n)
    t2 = torch.cross(n, t1, dim=-1)
    t2 = t2 / (t2.norm(dim=-1, keepdim=True) + 1e-8)
    return t1, t2, n


def _compute_vertex_normals_from_mesh(faces: np.ndarray, verts_np: np.ndarray) -> np.ndarray:
    v0 = verts_np[faces[:, 0]]
    v1 = verts_np[faces[:, 1]]
    v2 = verts_np[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vert_normals = np.zeros_like(verts_np)
    np.add.at(vert_normals, faces[:, 0], face_normals)
    np.add.at(vert_normals, faces[:, 1], face_normals)
    np.add.at(vert_normals, faces[:, 2], face_normals)
    return vert_normals / (np.linalg.norm(vert_normals, axis=-1, keepdims=True) + 1e-8)


def _compute_vertex_neighbors(faces: np.ndarray, n_verts: int) -> list[list[int]]:
    neighbors = [set() for _ in range(n_verts)]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [sorted(nbrs) for nbrs in neighbors]


def _pick_frame_neighbors(vid: int, vertex_neighbors: list[list[int]], verts_np: np.ndarray, tree: KDTree) -> tuple[int, int]:
    neighbors = [nbr for nbr in vertex_neighbors[vid] if nbr != vid]
    if len(neighbors) >= 2:
        return neighbors[0], neighbors[1]

    _, nn_ids = tree.query(verts_np[vid], k=4)
    nn_ids = [int(nbr) for nbr in np.atleast_1d(nn_ids) if int(nbr) != vid][:2]
    if len(nn_ids) != 2:
        raise ValueError(f"Unable to find two neighbors for vertex {vid}")
    return nn_ids[0], nn_ids[1]


def _procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    h = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(h)
    sign_mat = np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)])
    rot = vt.T @ sign_mat @ u.T
    trans = tgt_mean - rot @ src_mean
    return rot.astype(np.float32), trans.astype(np.float32)


def _nanmean_with_all_nan(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = ~np.isnan(values)
    counts = valid.sum(axis=axis)
    safe_values = np.where(valid, values, 0.0)
    sums = safe_values.sum(axis=axis)
    mean = sums / np.maximum(counts, 1)
    mean[counts == 0] = np.nan
    return mean.astype(np.float32)
