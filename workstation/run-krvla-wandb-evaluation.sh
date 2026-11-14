#!/usr/bin/env bash
set -euo pipefail
upload_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$upload_repo/workstation/env.local.sh"
export NETRC="$KRVLA_WORK/config/wandb/netrc"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset WANDB_SERVICE
exec "$KRVLA_WORK/envs/train/bin/python" -u "$upload_repo/workstation/publish_krvla_wandb.py" \
  --request "$KRVLA_WORK/training/krv100-full-20261010/epoch1_table1_request.json" --watch
