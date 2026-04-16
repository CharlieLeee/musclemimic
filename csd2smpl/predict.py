"""Run a trained checkpoint on a split and save predicted SMPL params per file.

Output (one NPZ per source marker file)::

    pred_poses (T, 72)   axis-angle
    pred_betas (10,)     shape coefficients
    pred_trans (T, 3)    root translation (in normalised units)

Usage
-----
::

    python -m csd2smpl.predict \\
        --config csd2smpl/configs/v100_3way.yaml \\
        --ckpt   $HOME/csd2smpl/checkpoints/best.pt \\
        --split  test \\
        --out    $HOME/csd2smpl/predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from csd2smpl.config_utils import load_config
from csd2smpl.data.c3d_dataset import MarkerDataset
from csd2smpl.models.pipeline import Markers2SMPL


class _SequentialPerFileSampler(Sampler[int]):
    """Walk the dataset's window index in order so we can reassemble per-file."""

    def __init__(self, n_windows: int) -> None:
        self.n = n_windows

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self) -> int:
        return self.n


def _resolve_device(cfg: dict) -> torch.device:
    if "device" in cfg:
        return torch.device(cfg["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(cfg: dict, ckpt_path: Path, split: str, out_dir: Path) -> None:
    """Run model on ``split`` and save predicted SMPL params per source NPZ."""
    device = _resolve_device(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # No window overlap so reassembly is trivial.
    dataset = MarkerDataset(
        cfg["data_root"], split=split,
        seq_len=cfg["seq_len"], stride=cfg["seq_len"],
        cache_in_ram=cfg.get("cache_in_ram", False),
    )

    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"],
        sampler=_SequentialPerFileSampler(len(dataset)),
        num_workers=cfg.get("num_workers_val", 1),
        pin_memory=device.type == "cuda",
    )

    model = Markers2SMPL(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {ckpt_path} on {device}; {len(dataset)} windows over {len(dataset.files)} files")

    # Buffer predictions per source file; flush when we move to the next one.
    current_fi = -1
    poses_buf: list[np.ndarray] = []
    trans_buf: list[np.ndarray] = []
    betas_buf: list[np.ndarray] = []

    def _flush(fi: int) -> None:
        if fi < 0 or not poses_buf:
            return
        src = dataset.files[fi]
        rel = src.relative_to(Path(cfg["data_root"]))
        dst = out_dir / rel.with_suffix(".pred.npz")
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dst,
            pred_poses=np.concatenate(poses_buf, axis=0),
            pred_trans=np.concatenate(trans_buf, axis=0),
            pred_betas=np.mean(np.stack(betas_buf, axis=0), axis=0),
            source=str(rel),
        )
        poses_buf.clear()
        trans_buf.clear()
        betas_buf.clear()

    with torch.no_grad():
        widx = 0
        for batch in loader:
            markers = batch["markers"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            pp, pb, pt = model(markers, mask)
            pp_np = pp.detach().cpu().numpy()
            pb_np = pb.detach().cpu().numpy()
            pt_np = pt.detach().cpu().numpy()

            for b in range(pp_np.shape[0]):
                fi, _ = dataset.index[widx]
                if fi != current_fi and current_fi >= 0:
                    _flush(current_fi)
                current_fi = fi
                poses_buf.append(pp_np[b])
                trans_buf.append(pt_np[b])
                betas_buf.append(pb_np[b])
                widx += 1
        _flush(current_fi)

    print(f"Wrote {len(list(out_dir.rglob('*.pred.npz')))} prediction files to {out_dir}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    predict(cfg, args.ckpt, args.split, args.out)


if __name__ == "__main__":
    main()
