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
from csd2smpl.data.marker_layouts import get_layout, layout_to_arrays
from csd2smpl.data.synthesize import (
    amass_to_smpl72,
    joints_to_markers,
    synthesize_file,
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
) -> np.ndarray:
    """Deterministic SMPL joints derived from poses for real signal-to-fit.

    Each joint j is placed at ``trans + 0.1 * j * axis_angle_of_joint_j`` so
    joint positions encode the pose. Lets a learner actually drive loss down.
    """
    t = poses_smpl72.shape[0]
    poses_reshaped = poses_smpl72.reshape(t, 24, 3)
    joint_scales = 0.1 * np.arange(1, 25, dtype=np.float32).reshape(1, 24, 1)
    return (poses_reshaped * joint_scales + trans[:, None, :]).astype(np.float32)


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

    print("[1/3] amass_to_smpl72 padding ...", end=" ", flush=True)
    test_amass_to_smpl72_pads_hand_joints()
    print("OK")

    print("[2/3] joints_to_markers (CMU 41) shape + anchors ...", end=" ", flush=True)
    test_joints_to_markers_shape_and_anchors()
    print("OK")

    print("[3/3] fake-AMASS → synthesize → MarkerDataset → train step ...", flush=True)
    with tempfile.TemporaryDirectory() as td:
        test_synthesize_roundtrip_and_train_step(Path(td))
    print("OK")

    print("\nAMASS roundtrip smoke PASSED.")


if __name__ == "__main__":
    main()
