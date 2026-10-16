#!/usr/bin/env bash
set -euo pipefail
epoch_eval_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$epoch_eval_repo/workstation/env.local.sh"
export NETRC="$KRVLA_WORK/config/wandb/netrc"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$KRVLA_WORK/envs/train/bin/python" -u "$epoch_eval_repo/workstation/krvla_table1.py" \
    --request "$KRVLA_WORK/training/krv100-full-20261010/epoch1_table1_request.json"
