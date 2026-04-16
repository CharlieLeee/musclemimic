"""End-to-end smoke: fake AMASS NPZ → synthesize markers → MarkerDataset → train step.

Bypasses smplx by injecting ``forward_fn`` into :func:`synthesize_file`,
so the test runs on bare checkouts without SMPL .pkl body-model files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from csd2smpl.data.c3d_dataset import MarkerDataset
from csd2smpl.data.marker_layouts import (
    CMU_TO_SSM_LABEL,
    get_layout,
    layout_to_arrays,
    load_cmu41_with_vertex_ids,
    load_ssm_placements,
)
from csd2smpl.data.synthesize import (
    amass_to_smpl72,
    joints_to_markers,
    synthesize_file,
    vertices_to_markers,
)
from csd2smpl.losses import compute_losses
from csd2smpl.models.pipeline import Markers2SMPL


SMOKE_CFG = Path(__file__).parents[1] / "configs" / "smoke.yaml"


def make_fake_amass_npz(path: Path, n_frames: int = 120, fps: float = 120.0) -> None:
    """Write a file matching the AMASS NPZ schema (poses 156, betas 16, etc.)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.savez(
        path,
        poses=(0.1 * rng.standard_normal((n_frames, 156))).astype(np.float32),
        betas=rng.standard_normal(16).astype(np.float32),
        trans=(0.01 * rng.standard_normal((n_frames, 3))).astype(np.float32),
        gender=np.array("neutral"),
        mocap_framerate=np.float32(fps),
    )


def stub_forward(
    poses_smpl72: np.ndarray,
    betas10: np.ndarray,
    trans: np.ndarray,
    model_path: str,
    gender: str = "neutral",
    device: str = "cpu",
    batch_size: int = 256,
    return_vertices: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Deterministic joints (+ optional vertices) derived from poses.

    Each joint j is placed at ``trans + 0.1 * j * axis_angle_of_joint_j`` so
    joint positions encode the pose. Vertices (when requested) are derived
    from joints with a fixed per-vertex offset; enough to smoke-test the
    vertex-mode code path without smplx.
    """
    t = poses_smpl72.shape[0]
    poses_reshaped = poses_smpl72.reshape(t, 24, 3)
    joint_scales = 0.1 * np.arange(1, 25, dtype=np.float32).reshape(1, 24, 1)
    joints = (poses_reshaped * joint_scales + trans[:, None, :]).astype(np.float32)
    if not return_vertices:
        return joints, None
    # Synthesize 6890 fake vertices so vertex-mode indexing is exercised.
    # Each vertex is a deterministic function of (joint, vertex_id).
    n_verts = 6890
    vert_offsets = (np.arange(n_verts, dtype=np.float32).reshape(1, n_verts, 1) * 1e-4)
    vertices = joints[:, :1, :] + vert_offsets
    return joints, vertices


def _fake_subset_tree(tmp_root: Path) -> tuple[Path, Path]:
    """Lay out ``<root>/<subset>/<session>.npz`` and return (in, out) roots."""
    amass_root = tmp_root / "amass"
    out_root = tmp_root / "amass_markers"
    # Use a sub-dataset name in AMASS_SPLITS["train"] so the loader picks it up.
    make_fake_amass_npz(amass_root / "CMU" / "01" / "01_01.npz", n_frames=256, fps=120.0)
    make_fake_amass_npz(amass_root / "CMU" / "02" / "02_01.npz", n_frames=200, fps=60.0)
    return amass_root, out_root


def test_amass_to_smpl72_pads_hand_joints() -> None:
    """AMASS→SMPL-24 keeps body 66 dims and zero-pads 6 more."""
    poses = np.arange(156, dtype=np.float32).reshape(1, 156)
    converted = amass_to_smpl72(poses)
    assert converted.shape == (1, 72)
    np.testing.assert_array_equal(converted[0, :66], poses[0, :66])
    np.testing.assert_array_equal(converted[0, 66:], np.zeros(6, dtype=np.float32))


def test_joints_to_markers_shape_and_anchors() -> None:
    """Every CMU 41 marker is anchored at its joint + offset."""
    layout = get_layout("cmu_41")
    anchors, offsets, _ = layout_to_arrays(layout)

    t = 5
    joints = np.zeros((t, 24, 3), dtype=np.float32)
    # Make each joint position = its index (broadcast along xyz) so we can
    # verify the gather picks the right rows.
    joints[:] = np.arange(24, dtype=np.float32).reshape(1, 24, 1)

    markers = joints_to_markers(joints, layout)
    assert markers.shape == (t, 41, 3)
    # Each marker[t, m] = anchors[m] + offsets[m]
    expected = anchors[None, :, None].astype(np.float32) + offsets[None, :, :]
    np.testing.assert_allclose(markers, np.broadcast_to(expected, (t, 41, 3)), atol=1e-5)


def test_cmu_to_ssm_label_covers_all_41() -> None:
    """Every CMU 41 label has an entry in the SSM alias table (None or str)."""
    layout = get_layout("cmu_41")
    names = [s.name for s in layout]
    missing = [n for n in names if n not in CMU_TO_SSM_LABEL]
    assert not missing, f"CMU 41 labels missing from CMU_TO_SSM_LABEL: {missing}"


def test_load_cmu41_with_vertex_ids_uses_ssm_json() -> None:
    """If the SSM JSON is fetched, loading populates vertex_ids on layout."""
    ssm_path = Path(__file__).parents[1] / "data" / "external" / "ssm_all_marker_placements.json"
    if not ssm_path.exists():
        import pytest
        pytest.skip(f"{ssm_path} not present; run scripts/fetch_external.sh")

    placements = load_ssm_placements(ssm_path)
    assert len(placements) > 0
    first_trial = next(iter(placements))
    assert "LFHD" in placements[first_trial] and "RFHD" in placements[first_trial]

    layout = load_cmu41_with_vertex_ids(ssm_path, trial=first_trial)
    assert len(layout) == 41
    populated = [s for s in layout if s.vertex_id is not None]
    # With the current alias table, every CMU 41 label has an SSM equivalent.
    assert len(populated) == 41, (
        f"expected all 41 CMU markers to map to SSM; got {len(populated)}"
    )
    for s in populated:
        assert 0 <= s.vertex_id < 6890, f"{s.name}: vertex_id={s.vertex_id} out of range"


def test_vertices_to_markers_uses_vertex_ids_and_falls_back() -> None:
    """Markers with vertex_id use mesh; those without fall back to joints."""
    from dataclasses import replace
    from csd2smpl.data.marker_layouts import CMU_41

    # Build a mini layout of 3 markers: 2 with vertex_id, 1 without.
    layout = (
        replace(CMU_41[0], vertex_id=10),
        replace(CMU_41[1], vertex_id=20),
        replace(CMU_41[2], vertex_id=None),   # fallback path
    )

    t = 4
    joints = np.zeros((t, 24, 3), dtype=np.float32)
    joints[:] = np.arange(24, dtype=np.float32).reshape(1, 24, 1)
    vertices = np.zeros((t, 6890, 3), dtype=np.float32)
    # vertices[:, k, :] = [k, k, k] so we can check the gather.
    vertices[:] = np.arange(6890, dtype=np.float32).reshape(1, 6890, 1)

    out = vertices_to_markers(vertices, joints, layout)
    assert out.shape == (t, 3, 3)
    # Marker 0: from vertex 10
    np.testing.assert_allclose(out[:, 0, :], 10.0)
    # Marker 1: from vertex 20
    np.testing.assert_allclose(out[:, 1, :], 20.0)
    # Marker 2: joint-anchored (no vertex_id) → joints[anchor] + offset
    anchor = CMU_41[2].anchor_joint
    offset = np.asarray(CMU_41[2].offset, dtype=np.float32)
    expected = np.full((t, 3), float(anchor)) + offset
    np.testing.assert_allclose(out[:, 2, :], expected, atol=1e-5)


def test_synthesize_roundtrip_and_train_step(tmp_path: Path) -> None:
    """Fake AMASS → synthesize markers → MarkerDataset → one training step with loss drop."""
    amass_root, out_root = _fake_subset_tree(tmp_path)

    for src in sorted(amass_root.rglob("*.npz")):
        rel = src.relative_to(amass_root)
        dst = out_root / rel.with_suffix(".markers.npz")
        wrote = synthesize_file(
            npz_path=src,
            out_path=dst,
            model_path="unused",
            layout_name="cmu_41",
            noise_std=0.0,
            dropout_p=0.0,
            target_fps=30.0,
            forward_fn=stub_forward,
        )
        assert wrote, f"synthesize_file refused to write {src}"

    # Verify on-disk schema matches the MarkerDataset contract.
    sample_path = next(out_root.rglob("*.markers.npz"))
    with np.load(sample_path) as out:
        assert out["markers"].shape[1:] == (41, 3)
        assert out["poses"].shape[1] == 72
        assert out["betas"].shape == (10,)
        assert out["trans"].shape[1] == 3
        # 120 Hz → 30 Hz: keep every 4th of 256 = 64 frames.
        assert out["markers"].shape[0] == 64
        assert str(out["layout"]) == "cmu_41"
        assert "marker_names" in out.files

    cfg = _load_cfg()
    ds = MarkerDataset(
        out_root, split="train",
        seq_len=cfg["seq_len"], stride=cfg["seq_len"],
    )
    assert len(ds) > 0, "dataset produced zero windows"

    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    batch = next(iter(loader))
    assert tuple(batch["markers"].shape[1:]) == (cfg["seq_len"], cfg["n_markers"], 3)
    assert tuple(batch["poses_gt"].shape[1:]) == (cfg["seq_len"], cfg["n_smpl_joints"] * 3)
    assert batch["betas_gt"].shape[1] == cfg["n_betas"]

    torch.manual_seed(0)
    model = Markers2SMPL(cfg)
    opt = AdamW(model.parameters(), lr=cfg["lr"])
    kw = {
        "w_pose": cfg["w_pose"], "w_shape": cfg["w_shape"],
        "w_trans": cfg["w_trans"], "w_smooth": cfg["w_smooth"],
    }

    model.train()

    def step_loss() -> torch.Tensor:
        pred = model(batch["markers"], batch["mask"])
        return compute_losses(
            *pred, batch["poses_gt"], batch["betas_gt"], batch["trans_gt"], **kw,
        )["loss"]

    initial = step_loss().item()
    for _ in range(20):
        opt.zero_grad()
        loss = step_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    final = step_loss().item()
    assert final < initial, (
        f"expected loss to drop after 20 steps; "
        f"initial={initial:.4f}, final={final:.4f}"
    )


def _load_cfg() -> dict:
    with SMOKE_CFG.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run tests against an explicit tmp dir so the script is standalone."""
    import tempfile

    print("[1/6] amass_to_smpl72 padding ...", end=" ", flush=True)
    test_amass_to_smpl72_pads_hand_joints()
    print("OK")

    print("[2/6] joints_to_markers (CMU 41) shape + anchors ...", end=" ", flush=True)
    test_joints_to_markers_shape_and_anchors()
    print("OK")

    print("[3/6] CMU→SSM alias table covers all 41 ...", end=" ", flush=True)
    test_cmu_to_ssm_label_covers_all_41()
    print("OK")

    print("[4/6] load_cmu41_with_vertex_ids from SSM JSON ...", end=" ", flush=True)
    try:
        test_load_cmu41_with_vertex_ids_uses_ssm_json()
        print("OK")
    except BaseException as exc:
        if "skip" in str(exc).lower() or "Skipped" in exc.__class__.__name__:
            print(f"SKIP ({exc})")
        else:
            raise

    print("[5/6] vertices_to_markers uses vertex IDs + fallback ...", end=" ", flush=True)
    test_vertices_to_markers_uses_vertex_ids_and_falls_back()
    print("OK")

    print("[6/6] fake-AMASS → synthesize → MarkerDataset → train step ...", flush=True)
    with tempfile.TemporaryDirectory() as td:
        test_synthesize_roundtrip_and_train_step(Path(td))
    print("OK")

    print("\nAMASS roundtrip smoke PASSED.")


if __name__ == "__main__":
    main()
