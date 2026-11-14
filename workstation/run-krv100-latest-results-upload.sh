#!/usr/bin/env bash
set -euo pipefail
latest_upload_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$latest_upload_repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$KRVLA_WORK/envs/train/bin/python" -u "$latest_upload_repo/workstation/upload_krv100_progress.py" --watch \
  --results "$KRVLA_WORK/results/krv100-latest-epoch008p09-task-table1-20261025" \
  --state-dir "$KRVLA_WORK/training/krv100-20k-from-epoch003-20261021/wandb-latest-checkpoint" \
  --report-prefix reports/task-subset/epoch008p09
