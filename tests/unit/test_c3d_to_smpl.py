from __future__ import annotations

import numpy as np
import pytest

from musclemimic.web_viewer import c3d_to_smpl


def test_build_marker_layout_matches_moshpp_aliases() -> None:
    layout = c3d_to_smpl.build_marker_layout(
        ["LASI", "RASI", "LPSI", "RPSI", "LWRA", "LWRB", "C7"],
        wrist_markers_on_stick=True,
    )

    assert layout.labels == ["C7", "LBWT", "LFWT", "LIWR", "LOWR", "RBWT", "RFWT"]
    assert layout.marker_vids["LFWT"] == 857
    assert layout.marker_vids["RBWT"] == 6544
    assert layout.marker_vids["LIWR"] == 2112
    assert layout.marker_type["LIWR"] == "wrist"
    assert layout.marker_type["LOWR"] == "wrist"
    assert layout.marker_type["LFWT"] == "body"
    assert layout.m2b_distance == {"body": 0.0095, "wrist": 0.039}


def test_prepare_marker_observations_collapses_duplicate_aliases() -> None:
    positions = np.array(
        [
            [
                [1.0, 0.0, 0.0],  # LASI -> LFWT
                [3.0, 0.0, 0.0],  # duplicate LFWT
                [10.0, 0.0, 0.0],  # unknown
                [np.nan, np.nan, np.nan],  # LWRB -> LOWR
                [5.0, 1.0, 1.0],  # duplicate LOWR
            ]
        ],
        dtype=np.float32,
    )

    prepared = c3d_to_smpl.prepare_marker_observations(
        positions,
        ["LASI", "LFWT", "UNKNOWN", "LWRB", "LOWR"],
    )

    assert prepared.labels == ["LFWT", "LOWR"]
    np.testing.assert_allclose(prepared.positions[0, 0], np.array([2.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(prepared.positions[0, 1], np.array([5.0, 1.0, 1.0], dtype=np.float32))
    assert prepared.unknown_labels == ["UNKNOWN"]


def test_pick_stagei_frames_random_strict_filters_by_availability() -> None:
    markers = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[np.nan, np.nan, np.nan], [2.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    picks = c3d_to_smpl.pick_stagei_frames(markers, num_frames=2, seed=123, least_avail_markers=1.0, strict=True)
    assert picks.tolist() == [1, 3]

    with pytest.raises(ValueError, match="Not enough frames"):
        c3d_to_smpl.pick_stagei_frames(markers, num_frames=3, seed=123, least_avail_markers=1.0, strict=True)


def test_stage_weight_formulas_match_moshpp_defaults() -> None:
    stage1 = c3d_to_smpl.compute_stagei_weights(num_markers=23, marker_types=["body", "wrist"], anneal_factor=0.5)
    assert stage1.data == pytest.approx((75.0 / 0.5) * (46.0 / 23.0))
    assert stage1.pose_body == pytest.approx(3.0 * 0.5)
    assert stage1.betas == pytest.approx(10.0 * 0.5)
    assert stage1.init_by_type["body"] == pytest.approx(300.0 * 0.5)
    assert stage1.surf == pytest.approx(10000.0)

    stage2 = c3d_to_smpl.compute_stageii_weights(num_observed_markers=18, num_total_markers=24)
    assert stage2.anneal_factor == pytest.approx(1.0 + ((24 - 18) / 24.0) * 2.5)
    assert stage2.data == pytest.approx(400.0 * (46.0 / 18.0))
    assert stage2.pose_body == pytest.approx(1.6 * stage2.anneal_factor)
    assert stage2.velo == pytest.approx(2.5)


def test_load_c3d_markers_handles_units_and_missing_points_without_axis_remap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_c3d = {
        "parameters": {
            "POINT": {
                "LABELS": {"value": ["A", "B"]},
                "UNITS": {"value": ["mm"]},
            }
        },
        "header": {"points": {"frame_rate": 120}},
        "data": {
            "points": np.array(
                [
                    [[1000.0], [0.0]],
                    [[2000.0], [0.0]],
                    [[3000.0], [0.0]],
                    [[0.0], [0.0]],
                ],
                dtype=np.float32,
            ),
            "meta_points": {
                "residuals": np.array([[[1.0], [-1.0]]], dtype=np.float32),
            },
        },
    }

    class FakeEzc3d:
        @staticmethod
        def c3d(_: str):
            return fake_c3d

    monkeypatch.setattr(c3d_to_smpl, "ezc3d", FakeEzc3d)
    positions, labels, fps = c3d_to_smpl.load_c3d_markers("dummy.c3d")

    assert labels == ["A", "B"]
    assert fps == 120.0
    np.testing.assert_allclose(positions[0, 0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.isnan(positions[0, 1]).all()


def test_fit_smpl_to_c3d_cached_uses_existing_cache(tmp_path) -> None:
    cache_dir = tmp_path / ".smpl_cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "sample_smpl.npz"
    np.savez(
        cache_path,
        pose_aa=np.ones((2, 72), dtype=np.float32),
        trans=np.ones((2, 3), dtype=np.float32),
        betas=np.arange(10, dtype=np.float32),
        fps=np.array(120.0, dtype=np.float32),
        gender=np.array("neutral"),
    )

    result = c3d_to_smpl.fit_smpl_to_c3d_cached("sample.c3d", "smpl_dir", cache_dir=str(cache_dir))

    assert result["pose_aa"].shape == (2, 72)
    assert result["trans"].shape == (2, 3)
    np.testing.assert_array_equal(result["betas"], np.arange(10, dtype=np.float32))
    assert result["fps"] == 120.0
    assert result["gender"] == "neutral"


def test_save_motion_data_as_amass_smplh_npz_expands_pose_to_156(tmp_path) -> None:
    output_path = tmp_path / "motion_smplh.npz"
    source = {
        "pose_aa": np.ones((3, 72), dtype=np.float32),
        "trans": np.arange(9, dtype=np.float32).reshape(3, 3),
        "betas": np.arange(16, dtype=np.float32),
        "fps": 30.0,
        "gender": "neutral",
    }

    written = c3d_to_smpl.save_motion_data_as_amass_smplh_npz(source, output_path)
    data = np.load(written, allow_pickle=True)

    assert written == output_path
    assert data["poses"].shape == (3, 156)
    np.testing.assert_array_equal(data["poses"][:, :72], np.ones((3, 72), dtype=np.float32))
    np.testing.assert_array_equal(data["poses"][:, 72:], np.zeros((3, 84), dtype=np.float32))
    np.testing.assert_array_equal(data["trans"], source["trans"])
    np.testing.assert_array_equal(data["betas"], source["betas"])
    assert float(data["mocap_framerate"]) == 30.0
    assert str(data["gender"]) == "neutral"
