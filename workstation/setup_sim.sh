#!/usr/bin/env bash
# Run after activating a dedicated Python 3.11 simulator conda environment.
set -euo pipefail
: "${ISAACLAB_ROOT:?Source workstation/env.local.sh first}"
python - <<'PY'
import platform, sys
if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 11):
    raise SystemExit('Activate a dedicated Linux x86_64 Python 3.11 simulator environment.')
libc, version = platform.libc_ver()
if libc != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 35):
    raise SystemExit('Isaac Sim pip installation requires glibc >=2.35; use the NVIDIA binary installation guide.')
PY
if test -e "$ISAACLAB_ROOT"; then
  test "$(git -C "$ISAACLAB_ROOT" rev-parse HEAD)" = 37ddf626871758333d6ed89cf64ad702aef127d0 || { echo 'Existing Isaac Lab is not v2.3.2; choose a new ISAACLAB_ROOT.'; exit 1; }
  test -z "$(git -C "$ISAACLAB_ROOT" status --porcelain --untracked-files=no)" || { echo 'Existing Isaac Lab has edits; preserve it and choose a new directory.'; exit 1; }
else
  git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_ROOT"
fi
python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
bash "$ISAACLAB_ROOT/isaaclab.sh" --install none
python -m pip install shapely==2.1.2 pyzmq==26.4.0 h5py
# Recheck CUDA build after Isaac Lab installs its dependencies.
python - <<'PY'
import torch
if torch.version.cuda != '12.8':
    raise SystemExit('Simulator torch changed CUDA builds; inspect install log before proceeding.')
x = torch.ones(4, device='cuda')
assert (x + x).tolist() == [2.0]*4
torch.cuda.synchronize()
print('Simulator CUDA kernel check passed:', torch.cuda.get_device_name(0))
PY
