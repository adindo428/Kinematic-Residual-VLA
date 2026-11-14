#!/usr/bin/env bash
set -euo pipefail
task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$task_repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc" HF_LEROBOT_HOME="$HF_HOME/lerobot"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
exec "$KRVLA_WORK/envs/train/bin/python" -u "$task_repo/workstation/task_aligned_table1.py" run \
  --root "$KRVLA_WORK/results/krv100-epoch010-task-table1-20261024"
