"""Dataset that reads synthesized (or real) marker NPZ files.

Contract — every ``__getitem__`` returns a dict with exactly these keys::

    markers   (T, M, 3)   float32   — root-centred & height-normalised markers
    mask      (T, M)      bool      — True = valid marker
    poses_gt  (T, 72)     float32   — SMPL axis-angle target
    betas_gt  (10,)       float32   — SMPL shape target
    trans_gt  (T, 3)      float32   — root translation (in normalised units)

``M`` is the marker count of the layout used during synthesis (e.g. 41 for
CMU). It must match ``cfg["n_markers"]`` at training time.

Real-C3D adapters resample to the same layout (or one with a registered
mapping) before fulfilling the contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# Standard AMASS train/val/test sub-dataset split (Mahmood et al. 2019).
# Paths under ``data_root`` are expected to start with the sub-dataset name,
# e.g. ``<root>/CMU/01/01_01.markers.npz``.
AMASS_SPLITS: dict[str, list[str]] = {
    "train": [
        "ACCAD", "BMLhandball", "BMLmovi", "BioMotionLab_NTroje",
        "CMU", "DFaust_67", "EKUT", "Eyes_Japan_Dataset",
        "KIT", "MPI_Limits", "SSM_synced", "TCD_handMocap",
        "TotalCapture", "Transitions_mocap",
    ],
    "val": ["MPI_HDM05", "HumanEva"],
    "test": ["SFU", "MPI_mosh"],
}


# Files written by csd2smpl.data.synthesize use this suffix.
DATASET_SUFFIX = ".markers.npz"


class MarkerDataset(Dataset):
    """Read synthesized marker NPZs and yield fixed-length training windows."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        seq_len: int = 64,
        stride: int = 32,
        splits: dict[str, list[str]] | None = None,
        cache_in_ram: bool = True,
        suffix: str = DATASET_SUFFIX,
    ) -> None:
        """Build the window index over all marker NPZs in ``root``.

        Parameters
        ----------
        root : str or Path
            Directory produced by :mod:`csd2smpl.scripts.synthesize_dataset`.
        split : str
            'train' | 'val' | 'test' (uses :data:`AMASS_SPLITS` by default).
        seq_len : int
            Number of frames per training window.
        stride : int
            Hop between windows. ``stride < seq_len`` → overlapping.
        splits : dict, optional
            Override the sub-dataset split table. Useful for tests or for
            real-C3D corpora that don't use AMASS naming.
        cache_in_ram : bool
            Cache loaded NPZs in a dict keyed by file index. Fine for the
            synthesised dataset (~few hundred MB); turn off for huge corpora.
        suffix : str
            File suffix to scan for. Defaults to ``.markers.npz``.
        """
        self.seq_len = seq_len
        self.stride = stride
        self.cache_in_ram = cache_in_ram
        self._cache: dict[int, dict[str, np.ndarray]] = {}

        splits = splits or AMASS_SPLITS
        if split not in splits:
            raise ValueError(
                f"split={split!r} not in splits keys {sorted(splits)}"
            )
        allowed = set(splits[split])

        root = Path(root)
        # rglob ``*<suffix>`` handles both ``.markers.npz`` and overrides.
        all_files = sorted(root.rglob(f"*{suffix}"))
        self.files: list[Path] = [
            f for f in all_files
            if len(f.relative_to(root).parts) > 0
            and f.relative_to(root).parts[0] in allowed
        ]

        self.index: list[tuple[int, int]] = []
        for fi, path in enumerate(self.files):
            with np.load(path) as seq:
                t = seq["markers"].shape[0]
            for start in range(0, t - seq_len + 1, stride):
                self.index.append((fi, start))

        print(
            f"MarkerDataset[{split}]: {len(self.files)} files, "
            f"{len(self.index)} windows, root={root}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, file_idx: int) -> dict[str, np.ndarray]:
        if self.cache_in_ram and file_idx in self._cache:
            return self._cache[file_idx]
        with np.load(self.files[file_idx]) as seq:
            data = {k: seq[k] for k in seq.files}
        if self.cache_in_ram:
            self._cache[file_idx] = data
        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fi, start = self.index[idx]
        seq = self._load(fi)
        end = start + self.seq_len

        markers_raw = seq["markers"][start:end].astype(np.float32)
        mask = seq["mask"][start:end].astype(bool)
        poses_gt = seq["poses"][start:end].astype(np.float32)
        betas_gt = seq["betas"].astype(np.float32)
        trans_gt = seq["trans"][start:end].astype(np.float32)

        # Root-centre using the per-frame mean of valid markers (no joint
        # available at inference). Translation target is shifted by the same
        # offset so the prediction stays consistent.
        # Use mask-broadcast to ignore zeroed-out (occluded) markers.
        m = mask[..., None].astype(np.float32)              # (T, M, 1)
        denom = m.sum(axis=1).clip(min=1e-6)                # (T, 1)
        centroid = (markers_raw * m).sum(axis=1) / denom    # (T, 3)
        markers = markers_raw - centroid[:, None, :]
        trans = trans_gt - centroid

        # Height-normalise using only valid markers' y-extent.
        ys = markers[..., 1]
        valid = ys[mask]
        if valid.size > 0:
            height = float(np.percentile(valid, 95) - np.percentile(valid, 5))
        else:
            height = 0.0
        height = max(height, 1e-3)
        markers = markers / height
        trans = trans / height

        return {
            "markers": torch.from_numpy(markers),
            "mask": torch.from_numpy(mask),
            "poses_gt": torch.from_numpy(poses_gt),
            "betas_gt": torch.from_numpy(betas_gt),
            "trans_gt": torch.from_numpy(trans),
        }


# Backwards-compatible alias for the old name in case anything imports it.
C3DDataset = MarkerDataset
