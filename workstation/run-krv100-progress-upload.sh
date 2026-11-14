#!/usr/bin/env bash
set -euo pipefail
upload_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$upload_repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$KRVLA_WORK/envs/train/bin/python" -u "$upload_repo/workstation/upload_krv100_progress.py" --watch
