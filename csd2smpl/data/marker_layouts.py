"""Optical-mocap marker layouts and their attachment to the SMPL skeleton.

The CMU 41-marker layout used here is the de-facto standard in the
SMPL/AMASS-driven RL/imitation-learning ecosystem (CMU MoCap source data,
AMP, MoCapAct). Marker label conventions follow the CMU MoCap database.

Marker → SMPL attachment
------------------------
Two paths:

1. **Vertex index (v2, preferred for real training)** — each marker is
   placed at a specific vertex of the 6890-vertex SMPL mesh. Loaded from
   the SSM mapping published by the AMASS authors (``nghorbani/amass``).
   Fetched at setup time via ``scripts/fetch_external.sh`` — see
   :func:`load_cmu41_with_vertex_ids`.

2. **Joint anchor + offset (v1 fallback)** — each marker is placed at
   ``smpl_joints[anchor_joint] + offset`` in the world frame. Loses the
   per-joint rotational signal but needs no SMPL mesh or external assets.
   Used by unit-test stubs and as a fallback when the SSM JSON is absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MarkerSpec:
    """How a single named marker attaches to a SMPL body.

    Attributes
    ----------
    name : str
        Canonical marker label (e.g. ``"LFHD"``).
    anchor_joint : int
        Index into SMPL_24 (see :mod:`csd2smpl.data.joint_maps`). The marker
        is placed relative to this joint's world position.
    offset : tuple[float, float, float]
        World-frame ``(x, y, z)`` offset in metres applied to the joint
        position. Approximate; see module docstring.
    vertex_id : int | None
        SMPL mesh vertex index for the high-accuracy attachment path. ``None``
        until populated from a source like SOMA or MoCapAct.
    """

    name: str
    anchor_joint: int
    offset: tuple[float, float, float]
    vertex_id: int | None = None


# CMU 41-marker layout. Anchor joint indices reference SMPL_24 from
# csd2smpl.data.joint_maps. Offsets are world-frame approximations
# (left/right is X, up is Y, forward is Z, in metres).
CMU_41: tuple[MarkerSpec, ...] = (
    # Head (4)
    MarkerSpec("LFHD", anchor_joint=15, offset=(-0.06,  0.06,  0.05)),
    MarkerSpec("RFHD", anchor_joint=15, offset=( 0.06,  0.06,  0.05)),
    MarkerSpec("LBHD", anchor_joint=15, offset=(-0.06, -0.06,  0.05)),
    MarkerSpec("RBHD", anchor_joint=15, offset=( 0.06, -0.06,  0.05)),
    # Torso (5)
    MarkerSpec("C7",   anchor_joint=12, offset=( 0.00, -0.05,  0.00)),
    MarkerSpec("T10",  anchor_joint=6,  offset=( 0.00, -0.10,  0.00)),
    MarkerSpec("CLAV", anchor_joint=12, offset=( 0.00,  0.05,  0.00)),
    MarkerSpec("STRN", anchor_joint=6,  offset=( 0.00,  0.08,  0.00)),
    MarkerSpec("RBAK", anchor_joint=6,  offset=( 0.10, -0.10,  0.00)),
    # Arms — 12 markers (shoulder, upper arm, elbow, forearm, wrist x2 per side)
    MarkerSpec("LSHO", anchor_joint=16, offset=(-0.02,  0.00,  0.00)),
    MarkerSpec("RSHO", anchor_joint=17, offset=( 0.02,  0.00,  0.00)),
    MarkerSpec("LUPA", anchor_joint=16, offset=(-0.05, -0.10,  0.00)),
    MarkerSpec("RUPA", anchor_joint=17, offset=( 0.05, -0.10,  0.00)),
    MarkerSpec("LELB", anchor_joint=18, offset=(-0.02,  0.00,  0.00)),
    MarkerSpec("RELB", anchor_joint=19, offset=( 0.02,  0.00,  0.00)),
    MarkerSpec("LFRM", anchor_joint=18, offset=(-0.05, -0.10,  0.00)),
    MarkerSpec("RFRM", anchor_joint=19, offset=( 0.05, -0.10,  0.00)),
    MarkerSpec("LWRA", anchor_joint=20, offset=(-0.03,  0.00,  0.00)),
    MarkerSpec("LWRB", anchor_joint=20, offset=( 0.03,  0.00,  0.00)),
    MarkerSpec("RWRA", anchor_joint=21, offset=( 0.03,  0.00,  0.00)),
    MarkerSpec("RWRB", anchor_joint=21, offset=(-0.03,  0.00,  0.00)),
    # Hands (2)
    MarkerSpec("LFIN", anchor_joint=22, offset=( 0.00, -0.05,  0.00)),
    MarkerSpec("RFIN", anchor_joint=23, offset=( 0.00, -0.05,  0.00)),
    # Pelvis (4)
    MarkerSpec("LASI", anchor_joint=0,  offset=(-0.10,  0.10,  0.00)),
    MarkerSpec("RASI", anchor_joint=0,  offset=( 0.10,  0.10,  0.00)),
    MarkerSpec("LPSI", anchor_joint=0,  offset=(-0.05, -0.10,  0.00)),
    MarkerSpec("RPSI", anchor_joint=0,  offset=( 0.05, -0.10,  0.00)),
    # Legs main (10)
    MarkerSpec("LTHI", anchor_joint=1,  offset=(-0.05, -0.20,  0.00)),
    MarkerSpec("RTHI", anchor_joint=2,  offset=( 0.05, -0.20,  0.00)),
    MarkerSpec("LKNE", anchor_joint=4,  offset=(-0.05,  0.00,  0.00)),
    MarkerSpec("RKNE", anchor_joint=5,  offset=( 0.05,  0.00,  0.00)),
    MarkerSpec("LANK", anchor_joint=7,  offset=(-0.05,  0.00,  0.00)),
    MarkerSpec("RANK", anchor_joint=8,  offset=( 0.05,  0.00,  0.00)),
    MarkerSpec("LHEE", anchor_joint=7,  offset=( 0.00,  0.00, -0.05)),
    MarkerSpec("RHEE", anchor_joint=8,  offset=( 0.00,  0.00, -0.05)),
    MarkerSpec("LTOE", anchor_joint=10, offset=( 0.00,  0.05,  0.00)),
    MarkerSpec("RTOE", anchor_joint=11, offset=( 0.00,  0.05,  0.00)),
    # Leg extras (4): shin + 1st metatarsal head per side
    MarkerSpec("LSHN", anchor_joint=4,  offset=(-0.02, -0.20,  0.00)),
    MarkerSpec("RSHN", anchor_joint=5,  offset=( 0.02, -0.20,  0.00)),
    MarkerSpec("LMT1", anchor_joint=10, offset=(-0.03,  0.05,  0.00)),
    MarkerSpec("RMT1", anchor_joint=11, offset=( 0.03,  0.05,  0.00)),
)

assert len(CMU_41) == 41, f"CMU_41 must have 41 markers, has {len(CMU_41)}"


# Public registry — add new layouts here as needed.
LAYOUTS: dict[str, tuple[MarkerSpec, ...]] = {
    "cmu_41": CMU_41,
}


def get_layout(name: str) -> tuple[MarkerSpec, ...]:
    """Look up a marker layout by name; raises with the available options."""
    if name not in LAYOUTS:
        raise KeyError(
            f"unknown marker layout {name!r}; available: {sorted(LAYOUTS)}"
        )
    return LAYOUTS[name]


def layout_to_arrays(
    layout: tuple[MarkerSpec, ...],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pack a layout into anchor-index and offset arrays for fast vectorised use.

    Parameters
    ----------
    layout : tuple of MarkerSpec
        E.g. :data:`CMU_41`.

    Returns
    -------
    anchors : np.ndarray
        Shape ``(M,)`` int64, anchor SMPL joint indices.
    offsets : np.ndarray
        Shape ``(M, 3)`` float32, world-frame per-marker offsets in metres.
    names : list[str]
        Marker label per row, length ``M``.
    """
    anchors = np.array([s.anchor_joint for s in layout], dtype=np.int64)
    offsets = np.array([s.offset for s in layout], dtype=np.float32)
    names = [s.name for s in layout]
    return anchors, offsets, names


def layout_vertex_ids(layout: tuple[MarkerSpec, ...]) -> np.ndarray:
    """Return per-marker SMPL vertex indices as ``(M,)`` int64.

    Raises ``ValueError`` if any marker still has ``vertex_id=None``.
    """
    missing = [s.name for s in layout if s.vertex_id is None]
    if missing:
        raise ValueError(
            f"layout has unpopulated vertex_ids for: {missing}. "
            f"Run scripts/fetch_external.sh and use load_cmu41_with_vertex_ids."
        )
    return np.array([s.vertex_id for s in layout], dtype=np.int64)


# ── SSM (nghorbani/amass) → CMU name alias table ─────────────────────────
#
# The SSM JSON from AMASS uses slightly different marker labels than the
# CMU convention we use in :data:`CMU_41`. This table resolves CMU labels
# to their SSM equivalents. ``None`` means "no direct SSM equivalent — fall
# back to the joint-anchored offset for this marker". Wrist medial/lateral
# conventions: CMU's ``A`` = lateral (outer/radial), ``B`` = medial (inner/
# ulnar); SSM uses ``O/I`` (outer/inner).

CMU_TO_SSM_LABEL: dict[str, str] = {
    # Direct hits (CMU label == SSM label)
    "LFHD": "LFHD", "RFHD": "RFHD", "LBHD": "LBHD", "RBHD": "RBHD",
    "C7":   "C7",   "CLAV": "CLAV", "STRN": "STRN", "RBAK": "RBAK",
    "LSHO": "LSHO", "RSHO": "RSHO",
    "LUPA": "LUPA",  # RUPA → RUPA2 below
    "LELB": "LELB", "RELB": "RELB",
    "LFRM": "LFRM", "RFRM": "RFRM",
    "LFIN": "LFIN", "RFIN": "RFIN",
    "LASI": "LASI", "RASI": "RASI", "LPSI": "LPSI", "RPSI": "RPSI",
    "RTHI": "RTHI",
    "LKNE": "LKNE", "RKNE": "RKNE",
    "LSHN": "LSHN", "RSHN": "RSHN",
    "LANK": "LANK", "RANK": "RANK",
    "LHEE": "LHEE", "RHEE": "RHEE",
    "LTOE": "LTOE", "RTOE": "RTOE",
    "LMT1": "LMT1", "RMT1": "RMT1",

    # Aliases (CMU label ≠ SSM label)
    "T10":  "T8",      # adjacent thoracic vertebra
    "RUPA": "RUPA2",   # SSM uses RUPA2 for the right upper arm
    "LTHI": "LTHILO",  # SSM splits thigh into upper/lower; pick lower
    "LWRA": "LOWR",    # A = outer/radial
    "LWRB": "LIWR",    # B = inner/ulnar
    "RWRA": "ROWR",
    "RWRB": "RIWR",
}


def load_ssm_placements(path: str | Path) -> dict[str, dict[str, int]]:
    """Load the SSM marker→vertex JSON produced by nghorbani/amass.

    Parameters
    ----------
    path : str or Path
        Path to ``ssm_all_marker_placements.json``.

    Returns
    -------
    dict[str, dict[str, int]]
        Outer key = trial name, inner = marker label → SMPL vertex index.
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path} has unexpected structure (empty or non-dict)")
    for trial, markers in data.items():
        if not isinstance(markers, dict) or not markers:
            raise ValueError(f"trial {trial!r} has no marker entries")
    return data


def load_cmu41_with_vertex_ids(
    ssm_json_path: str | Path,
    trial: str | None = None,
) -> tuple[MarkerSpec, ...]:
    """Return a CMU 41 layout with ``vertex_id`` populated from the SSM JSON.

    Parameters
    ----------
    ssm_json_path : str or Path
        Path to ``ssm_all_marker_placements.json`` (fetched by
        ``scripts/fetch_external.sh``).
    trial : str or None
        SSM trial name to use as the canonical placement. ``None`` selects
        the first key in the JSON (deterministic: JSON is stored ordered).

    Returns
    -------
    tuple of MarkerSpec
        CMU 41 with ``vertex_id`` filled in for every marker whose CMU
        label has an SSM alias. Markers without an alias retain
        ``vertex_id=None`` and will fall back to joint-anchored placement.
    """
    placements = load_ssm_placements(ssm_json_path)
    if trial is None:
        trial = next(iter(placements))
    if trial not in placements:
        raise KeyError(
            f"trial {trial!r} not in SSM placements; available: "
            f"{list(placements)[:5]}..."
        )
    ssm = placements[trial]

    out: list[MarkerSpec] = []
    unresolved: list[str] = []
    for spec in CMU_41:
        ssm_label = CMU_TO_SSM_LABEL.get(spec.name)
        vid = ssm.get(ssm_label) if ssm_label else None
        if vid is None:
            unresolved.append(spec.name)
        out.append(replace(spec, vertex_id=vid))
    if unresolved:
        # Not fatal: those markers fall back to joint-anchored offsets.
        # Print so the user knows something's not canonical.
        print(
            f"[marker_layouts] {len(unresolved)}/{len(CMU_41)} markers have no "
            f"SSM vertex alias; falling back to joint-anchored: {unresolved}"
        )
    return tuple(out)
