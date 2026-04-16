"""Inspect ``csd2smpl/data/raw/`` without extracting anything.

Identifies and reports on AMASS sub-dataset tarballs, SMPL/SMPL-H archives,
DMPL archives, partial downloads, and loose NPZ/PKL files. Designed to run
on the cluster against a populated ``data/raw/`` directory before kicking
off extraction.

Usage
-----
::

    python -m csd2smpl.scripts.inspect_raw [--raw_dir PATH] [--peek]

If ``--peek`` is set, lists the first few entries inside each archive
(slow for big tarballs).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Iterable


# Patterns for AMASS tarballs (sub-dataset name = filename stem before .tar.*).
_AMASS_PATTERN = re.compile(
    r"^(ACCAD|BMLhandball|BMLmovi|BioMotionLab_NTroje|CMU|DFaust_67|"
    r"DanceDB|EKUT|EyesJapanDataset|Eyes_Japan_Dataset|HumanEva|KIT|"
    r"MPI_HDM05|MPI_Limits|MPI_mosh|SFU|SOMA|SSM_synced|TCD_handMocap|"
    r"TotalCapture|Transitions_mocap)\.tar\.(bz2|gz|xz)$",
    re.IGNORECASE,
)

# SMPL/SMPL-H/SMPL-X body-model archives (filenames vary; match common forms).
_BODY_MODEL_HINTS = (
    "smpl_python", "smplh_g", "smplh", "smplx",
    "smpl_male", "smpl_female", "smpl_neutral",
    "mano",
)

# DMPL (Dynamic Multi-person Linear) archives.
_DMPL_HINTS = ("dmpls",)


def _human(n: int) -> str:
    """Bytes → human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _classify(name: str) -> str:
    """Return one of: ``amass`` | ``smpl`` | ``dmpl`` | ``partial`` | ``other``."""
    base = name.lower()
    if base.endswith(".crdownload") or base.endswith(".part") or base.endswith(".tmp"):
        return "partial"
    if _AMASS_PATTERN.match(name):
        return "amass"
    if any(h in base for h in _BODY_MODEL_HINTS):
        return "smpl"
    if any(h in base for h in _DMPL_HINTS):
        return "dmpl"
    if base.endswith(".npz") or base.endswith(".pkl"):
        return "loose"
    return "other"


def _peek_archive(path: Path, n: int = 5) -> list[str]:
    """Return up to ``n`` entry names from a tar archive without extracting."""
    if not path.suffix.lower() in {".bz2", ".gz", ".xz", ".tgz", ".tbz2"} \
            and ".tar" not in path.name.lower():
        return []
    try:
        with tarfile.open(path, "r:*") as tf:
            return [m.name for _, m in zip(range(n), tf)]
    except (tarfile.TarError, EOFError, OSError) as exc:
        return [f"<unreadable: {exc.__class__.__name__}: {exc}>"]


def _validate_tar(path: Path) -> tuple[bool, str]:
    """Verify the tar headers parse end-to-end (catches truncated/corrupt archives)."""
    try:
        with tarfile.open(path, "r:*") as tf:
            count = sum(1 for _ in tf)
        return True, f"{count} entries"
    except (tarfile.TarError, EOFError, OSError) as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def inspect(raw_dir: Path, peek: bool = False, validate: bool = False) -> int:
    """Walk ``raw_dir`` and print a report. Returns ``0`` on clean state, ``1`` on issues.

    Parameters
    ----------
    raw_dir : Path
        Directory to walk (non-recursive — tarballs live at the top level).
    peek : bool
        If True, list a few entries from each AMASS tarball.
    validate : bool
        If True, parse every tarball's headers end-to-end. Slow but catches
        truncated downloads.
    """
    if not raw_dir.exists():
        print(f"error: raw_dir does not exist: {raw_dir}", file=sys.stderr)
        return 1

    files = sorted(p for p in raw_dir.iterdir() if p.is_file())
    if not files:
        print(f"warning: no files in {raw_dir}")
        return 1

    by_kind: dict[str, list[Path]] = {
        "amass": [], "smpl": [], "dmpl": [], "loose": [], "partial": [], "other": [],
    }
    for f in files:
        by_kind[_classify(f.name)].append(f)

    issues = 0
    print(f"── Raw data inventory: {raw_dir} ─────────────────────")
    print(f"Total files: {len(files)}, total size: {_human(sum(f.stat().st_size for f in files))}")
    print()

    def _section(title: str, paths: list[Path]) -> None:
        nonlocal issues
        if not paths:
            return
        total = sum(p.stat().st_size for p in paths)
        print(f"[{title}]  {len(paths)} files, {_human(total)}")
        for p in paths:
            size = _human(p.stat().st_size)
            line = f"  {p.name:50s} {size:>10s}"
            if validate and (p.suffix in {".bz2", ".gz", ".xz"} or p.name.endswith((".tbz2", ".tgz"))):
                ok, info = _validate_tar(p)
                tag = "OK " if ok else "BAD"
                line += f"  [{tag}] {info}"
                if not ok:
                    issues += 1
            print(line)
            if peek and title == "AMASS":
                for entry in _peek_archive(p, n=3):
                    print(f"      ↳ {entry}")
        print()

    _section("AMASS", by_kind["amass"])
    _section("SMPL/SMPL-H/SMPL-X body models", by_kind["smpl"])
    _section("DMPL", by_kind["dmpl"])
    _section("Loose NPZ/PKL", by_kind["loose"])
    if by_kind["partial"]:
        print("[PARTIAL DOWNLOADS]")
        for p in by_kind["partial"]:
            print(f"  {p.name:50s} {_human(p.stat().st_size):>10s}  -- still downloading?")
            issues += 1
        print()
    if by_kind["other"]:
        print("[UNRECOGNIZED]")
        for p in by_kind["other"]:
            print(f"  {p.name:50s} {_human(p.stat().st_size):>10s}")
        print()

    print("── Coverage ─────────────────────────────────────────")
    have_amass = bool(by_kind["amass"])
    have_smpl = bool(by_kind["smpl"]) or any(p.suffix == ".pkl" for p in by_kind["loose"])
    print(f"  AMASS sub-datasets : {len(by_kind['amass'])} archive(s) "
          f"({'OK' if have_amass else 'MISSING'})")
    print(f"  SMPL body models   : {'present' if have_smpl else 'MISSING — needed for synthesis + eval'}")
    print(f"  DMPLs (optional)   : {'present' if by_kind['dmpl'] else 'absent'}")
    if not have_amass:
        issues += 1
    if not have_smpl:
        issues += 1

    print()
    if issues == 0:
        print("Inventory clean.")
    else:
        print(f"Inventory has {issues} issue(s) — see above.")
    return 0 if issues == 0 else 1


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir", type=Path,
        default=Path(__file__).parents[1] / "data" / "raw",
        help="directory to inspect (default: csd2smpl/data/raw)",
    )
    parser.add_argument(
        "--peek", action="store_true",
        help="list first few entries inside each AMASS tarball",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="walk every archive's headers end-to-end (slow)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return inspect(args.raw_dir, peek=args.peek, validate=args.validate)


if __name__ == "__main__":
    sys.exit(main())
