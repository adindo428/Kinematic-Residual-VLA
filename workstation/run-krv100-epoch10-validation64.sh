#!/usr/bin/env bash
set -euo pipefail
validation_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$validation_repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc" HF_LEROBOT_HOME="$HF_HOME/lerobot"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
exec "$KRVLA_WORK/envs/train/bin/python" -u "$validation_repo/workstation/epoch10_heldout_evaluation.py" \
  --request "$KRVLA_WORK/results/krv100-epoch010-20k-validation64-20261024/request.json"
