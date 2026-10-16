#!/usr/bin/env bash
# Reinstall the verified paper-stack dependencies into the existing SSD profile.
# Miniforge and workstation/env.local.sh are bootstrapped once before this script.
set -euo pipefail
paper_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$paper_repo/workstation/env.local.sh"
case "$KRVLA_WORK" in
  /media/ailab-students/Linux_SSD/*) ;;
  *) echo 'KRVLA_WORK must be on the Linux_SSD disk.' >&2; exit 1 ;;
esac
for paper_env in policy sim45; do
  if [[ ! -x "$KRVLA_WORK/envs/$paper_env/bin/python" ]]; then
    "$CONDA_EXE" create --prefix "$KRVLA_WORK/envs/$paper_env" \
      --override-channels -c conda-forge python=3.10 pip -y
  fi
done
"$POLICY_PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
"$POLICY_PYTHON" -m pip install -r "$paper_repo/evaluation/official_dom/requirements-policy.txt" \
  -c "$paper_repo/evaluation/official_dom/requirements-constraints.txt"
"$SIM_PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
"$SIM_PYTHON" -m pip install -r "$paper_repo/workstation/requirements-build.txt"
"$SIM_PYTHON" -m pip install 'isaacsim[all,extscache]==4.5.0' \
  --extra-index-url https://pypi.nvidia.com -c "$paper_repo/workstation/requirements-sim45.txt"
test "$(git -C "$ISAACLAB_ROOT" rev-parse HEAD)" = 0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20
"$SIM_PYTHON" -m pip install -e "$ISAACLAB_ROOT/source/isaaclab" \
  -e "$ISAACLAB_ROOT/source/isaaclab_assets" -e "$ISAACLAB_ROOT/source/isaaclab_tasks" \
  shapely==2.1.2 pyzmq==26.4.0 h5py -r "$paper_repo/workstation/requirements-sim45.txt" \
  --build-constraint "$paper_repo/workstation/requirements-build.txt"
"$POLICY_PYTHON" -m pip check
"$SIM_PYTHON" -m pip check
