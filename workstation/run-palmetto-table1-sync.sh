#!/usr/bin/env bash
set -euo pipefail
sync_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$sync_repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$KRVLA_WORK/envs/train/bin/python" -u "$sync_repo/workstation/sync_palmetto_table1.py" --watch
