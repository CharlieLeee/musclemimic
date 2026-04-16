#!/usr/bin/env bash
# Set up a Python venv on the cluster for csd2smpl GPU training.
#
# Tested with Python 3.10/3.11. Installs CUDA 12.8 PyTorch wheels which
# run on CUDA 12.x and CUDA 13.x drivers via forward compatibility.
#
# Usage:
#   bash csd2smpl/scripts/setup_cluster.sh [venv_dir] [cuda_tag]
#
# Examples:
#   bash csd2smpl/scripts/setup_cluster.sh                 # ./.venv, cu128
#   bash csd2smpl/scripts/setup_cluster.sh ~/envs/csd cu124
#
# After it finishes:
#   source <venv_dir>/bin/activate
#   python -m csd2smpl.scripts.preflight
#   python -m csd2smpl.tests.test_smoke

set -euo pipefail

VENV_DIR="${1:-.venv}"
CUDA_TAG="${2:-cu128}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 not found on PATH" >&2
    exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
echo "Python: $PY_VERSION"
case "$PY_VERSION" in
    3.10|3.11|3.12) ;;
    *) echo "warning: csd2smpl is tested on 3.10–3.12, not $PY_VERSION" >&2 ;;
esac

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating venv at $VENV_DIR"
    # Some minimal Python 3.12 images ship venv without ensurepip; fall back
    # to --without-pip and bootstrap below.
    python3 -m venv "$VENV_DIR" 2>/dev/null \
        || python3 -m venv --without-pip "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Bootstrap pip if the venv lacks it (Debian/Ubuntu minimal Python 3.12).
if ! python -m pip --version >/dev/null 2>&1; then
    echo "pip not present in venv — bootstrapping ..."
    if python -m ensurepip --upgrade 2>/dev/null; then
        echo "  bootstrapped via ensurepip"
    else
        echo "  ensurepip unavailable; pulling get-pip.py"
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        python /tmp/get-pip.py
        rm -f /tmp/get-pip.py
    fi
fi

python -m pip install --upgrade pip wheel

echo ""
echo "Installing PyTorch ($CUDA_TAG) ..."
python -m pip install --index-url "https://download.pytorch.org/whl/$CUDA_TAG" torch

echo ""
echo "Installing csd2smpl deps ..."
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -r "$HERE/../requirements.txt"

echo ""
echo "── GPU check ─────────────────────────────────────────"
python - <<'PY'
import torch
print(f"torch     : {torch.__version__}")
print(f"cuda avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda build: {torch.version.cuda}")
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1024**3
        print(f"gpu[{i}]   : {p.name}  ({gb:.1f} GB, sm_{p.major}{p.minor})")
PY

echo ""
echo "Setup complete. Next steps:"
echo "  source $VENV_DIR/bin/activate"
echo "  python -m csd2smpl.scripts.preflight"
echo "  python -m csd2smpl.tests.test_smoke"
