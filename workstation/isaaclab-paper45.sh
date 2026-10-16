#!/usr/bin/env bash
# Select the SSD simulator Python without activating/deactivating the user's Conda.
set -euo pipefail
paper_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$paper_repo/workstation/env.local.sh"
export CONDA_PREFIX="$KRVLA_WORK/envs/sim45"
export CONDA_DEFAULT_ENV="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
if [[ "${TERM:-dumb}" == dumb ]]; then export TERM=xterm; fi
if [[ "${1:-}" == -p || "${1:-}" == --python ]]; then
  shift
  exec "$CONDA_PREFIX/bin/python" "$@"
fi
exec bash "$ISAACLAB_ROOT/isaaclab.sh" "$@"
