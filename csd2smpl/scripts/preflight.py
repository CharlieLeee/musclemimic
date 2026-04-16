"""Preflight: verify every prerequisite before launching long-running synthesis or training.

Checks
------
1. Python imports (torch, roma, numpy, yaml).
2. CUDA visibility (if running on GPU).
3. ``smplx`` import + SMPL .pkl files in the configured model dir.
4. AMASS root layout (at least one sub-dataset present, *_poses.npz files).
5. Markers output dir exists and is writable.
6. Tiny end-to-end forward/backward pass to catch driver/CUDA issues early.

Usage
-----
::

    python -m csd2smpl.scripts.preflight \\
        --amass_root /data/amass \\
        --smpl_dir   /data/body_models/smpl \\
        --out_root   /data/amass_markers
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Callable


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def check_imports() -> bool:
    print("[1] Python imports")
    needed = ["torch", "numpy", "yaml", "roma"]
    ok = True
    for name in needed:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            _ok(f"{name:8s} {ver}")
        except ImportError as exc:
            _fail(f"{name}: {exc}")
            ok = False
    return ok


def check_cuda(require_gpu: bool) -> bool:
    print("[2] CUDA")
    import torch

    avail = torch.cuda.is_available()
    if not avail:
        msg = f"torch.cuda.is_available()=False (built for CUDA {torch.version.cuda})"
        if require_gpu:
            _fail(msg)
            return False
        _warn(msg + " — continuing on CPU")
        return True

    _ok(f"torch {torch.__version__} (cuda build {torch.version.cuda})")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1024**3
        _ok(f"gpu[{i}] {p.name}  {gb:.1f} GB  sm_{p.major}{p.minor}")
    return True


def check_smplx(smpl_dir: Path | None) -> bool:
    print("[3] smplx + SMPL body models")
    try:
        import smplx  # noqa: F401
        _ok("smplx import OK")
    except ImportError as exc:
        _fail(f"smplx not installed: {exc}")
        _warn("install with: pip install 'smplx>=0.1.28'")
        return False

    if smpl_dir is None:
        _warn("--smpl_dir not provided; skipping body-model file check")
        return True

    if not smpl_dir.exists():
        _fail(f"smpl_dir does not exist: {smpl_dir}")
        return False

    expected = ["SMPL_NEUTRAL.pkl", "SMPL_MALE.pkl", "SMPL_FEMALE.pkl"]
    found = [f for f in expected if (smpl_dir / f).exists()]
    if not found:
        # smplx also accepts lowercase or under a smpl/ subdir.
        nested = smpl_dir / "smpl"
        if nested.is_dir():
            found = [f for f in expected if (nested / f).exists()]
            if found:
                _ok(f"found in {nested}: {found}")
                return True
        _fail(
            f"no SMPL .pkl files in {smpl_dir} "
            f"(expected one of {expected})"
        )
        _warn("download from https://smpl.is.tue.mpg.de/ (license-gated)")
        return False
    _ok(f"found in {smpl_dir}: {found}")
    return True


def check_amass(amass_root: Path | None, deep_validate: bool = False) -> bool:
    """Verify AMASS layout + sample-load a few NPZs to catch corruption."""
    print("[4] AMASS dataset")
    if amass_root is None:
        _warn("--amass_root not provided; skipping")
        return True
    if not amass_root.exists():
        _fail(f"amass_root does not exist: {amass_root}")
        return False

    from csd2smpl.data.c3d_dataset import AMASS_SPLITS
    expected = set(AMASS_SPLITS["train"] + AMASS_SPLITS["val"] + AMASS_SPLITS["test"])
    present = sorted(p.name for p in amass_root.iterdir() if p.is_dir())
    overlap = sorted(set(present) & expected)
    if not overlap:
        _fail(
            f"no canonical AMASS sub-datasets under {amass_root} "
            f"(found {present[:6]}{'...' if len(present) > 6 else ''})"
        )
        return False

    # AMASS subsets ship final *_poses.npz plus MoSh++ intermediates
    # (*_stagei.npz, *_stageii.npz, *_stageiii.npz) and per-subject shape
    # files. Only *_poses.npz follows the schema we care about; the rest
    # are discarded — mirrors synthesize_dataset.py's behaviour.
    def _poses_files(root: Path) -> list[Path]:
        poses = sorted(root.rglob("*_poses.npz"))
        if poses:
            return poses
        # Very old/custom subsets sometimes drop the suffix; fall back and
        # explicitly filter out known MoSh intermediates.
        all_npz = sorted(root.rglob("*.npz"))
        return [
            f for f in all_npz
            if not f.name.endswith(("_stagei.npz", "_stageii.npz", "_stageiii.npz"))
            and "shape" not in f.name.lower()
        ]

    npz_per_subset = {d: _poses_files(amass_root / d) for d in overlap}
    n_total = sum(len(v) for v in npz_per_subset.values())
    _ok(f"{len(overlap)} sub-datasets present, {n_total} *_poses.npz files total")
    for d, files in sorted(npz_per_subset.items()):
        marker = " (empty!)" if not files else ""
        print(f"      {d:25s} {len(files):5d} sequences{marker}")
        if not files:
            return False

    # Sample-load up to 3 files per subset and verify schema.
    import numpy as np
    from csd2smpl.data.synthesize import get_amass_framerate

    bad: list[tuple[Path, str]] = []
    sampled = 0
    for d, files in npz_per_subset.items():
        sample = files[:3] if not deep_validate else files
        for f in sample:
            sampled += 1
            try:
                with np.load(f, allow_pickle=True) as seq:
                    keys = set(seq.files)
                    missing = {"poses", "betas", "trans"} - keys
                    if missing:
                        bad.append((f, f"missing keys: {missing}"))
                        continue
                    poses = seq["poses"]
                    if poses.ndim != 2 or poses.shape[1] not in (66, 72, 156, 159, 165):
                        bad.append((f, f"unexpected poses shape: {poses.shape}"))
                        continue
                    fps = get_amass_framerate(seq)
                    if not 10 <= fps <= 240:
                        bad.append((f, f"suspicious fps: {fps}"))
            except Exception as exc:  # noqa: BLE001
                bad.append((f, f"{exc.__class__.__name__}: {exc}"))

    if bad:
        _fail(f"{len(bad)}/{sampled} sampled NPZs failed validation:")
        for f, why in bad[:5]:
            print(f"      {f.relative_to(amass_root)}  --  {why}")
        if len(bad) > 5:
            print(f"      ... and {len(bad) - 5} more")
        return False
    _ok(f"sampled {sampled} NPZs across all subsets, schema OK")
    return True


def check_out_root(out_root: Path | None) -> bool:
    print("[5] Markers output dir")
    if out_root is None:
        _warn("--out_root not provided; skipping")
        return True
    out_root.mkdir(parents=True, exist_ok=True)
    probe = out_root / ".csd2smpl_write_probe"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        _fail(f"{out_root} is not writable: {exc}")
        return False
    _ok(f"writable: {out_root}")
    return True


def check_arch_smoke() -> bool:
    print("[6] Mini forward/backward (sanity)")
    import torch
    from csd2smpl.losses import compute_losses
    from csd2smpl.models.pipeline import Markers2SMPL

    cfg = {
        "n_markers": 41, "n_smpl_joints": 24, "n_betas": 10,
        "d_model": 32, "n_heads": 2, "n_layers": 1,
        "dropout": 0.0, "max_seq_len": 64,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Markers2SMPL(cfg).to(device)
    markers = torch.randn(2, 8, 41, 3, device=device)
    mask = torch.ones(2, 8, 41, dtype=torch.bool, device=device)
    poses = torch.randn(2, 8, 72, device=device)
    betas = torch.randn(2, 10, device=device)
    trans = torch.randn(2, 8, 3, device=device)
    pp, pb, pt = model(markers, mask)
    loss = compute_losses(pp, pb, pt, poses, betas, trans)["loss"]
    loss.backward()
    if not torch.isfinite(loss):
        _fail(f"loss not finite: {loss.item()}")
        return False
    _ok(f"loss={loss.item():.4f}, gradients computed on {device}")
    return True


def main() -> int:
    """Run every check and exit non-zero if anything fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", type=Path, default=None)
    parser.add_argument("--smpl_dir", type=Path, default=None)
    parser.add_argument("--out_root", type=Path, default=None)
    parser.add_argument(
        "--require_gpu", action="store_true",
        help="fail if CUDA is not available",
    )
    parser.add_argument(
        "--deep_validate", action="store_true",
        help="open every AMASS NPZ to verify schema (slow)",
    )
    args = parser.parse_args()

    checks: list[tuple[str, Callable[[], bool]]] = [
        ("imports", check_imports),
        ("cuda", lambda: check_cuda(args.require_gpu)),
        ("smplx", lambda: check_smplx(args.smpl_dir)),
        ("amass", lambda: check_amass(args.amass_root, args.deep_validate)),
        ("out_root", lambda: check_out_root(args.out_root)),
        ("arch_smoke", check_arch_smoke),
    ]

    results: dict[str, bool] = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 — preflight should never crash
            _fail(f"{name} raised: {exc}")
            results[name] = False
        print()

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"{RED}PREFLIGHT FAILED:{RESET} {', '.join(failed)}")
        return 1
    print(f"{GREEN}PREFLIGHT OK — ready to synthesize + train.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
