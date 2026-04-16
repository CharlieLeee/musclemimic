"""Optical-mocap marker layouts and their attachment to the SMPL skeleton.

The CMU 41-marker layout used here is the de-facto standard in the
SMPL/AMASS-driven RL/imitation-learning ecosystem (CMU MoCap source data,
AMP, MoCapAct). Marker label conventions follow the CMU MoCap database.

Marker → SMPL attachment
------------------------
There are two ways to attach a marker to a SMPL body:

1. **Vertex index** — pick a specific vertex of the 6890-vertex SMPL mesh.
   Most accurate; this is what AMASS / MoSh++ / SOMA do internally. Vertex
   indices for common markersets are published in the SOMA and MoCapAct
   repos. **TODO: populate `vertex_id` fields below from those sources for
   real-data training; the current code path uses the joint-anchored
   approximation instead.**

2. **Joint anchor + offset** *(used by this v1 scaffold)* — each marker is
   placed at ``smpl_joints[anchor_joint] + offset`` in the world frame.
   Loses the per-joint rotational signal but is enough to drive the
   architecture end-to-end without requiring the SMPL .pkl body model
   for offset calibration. Offsets are anatomically rough (cm-scale).

Either path produces a ``(T, M, 3)`` array of synthetic marker positions
that the encoder consumes; switching paths is a one-line change in
:func:`csd2smpl.data.synthesize.joints_to_markers`.
"""

from __future__ import annotations

from dataclasses import dataclass

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
