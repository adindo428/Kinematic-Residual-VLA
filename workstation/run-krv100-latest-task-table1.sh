#!/usr/bin/env bash
set -euo pipefail
latest_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$latest_repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc" HF_LEROBOT_HOME="$HF_HOME/lerobot"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
exec "$KRVLA_WORK/envs/train/bin/python" -u "$latest_repo/workstation/latest_task_table1.py" \
  --root "$KRVLA_WORK/results/krv100-latest-epoch008p09-task-table1-20261025"
