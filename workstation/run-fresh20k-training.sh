#!/usr/bin/env bash
set -euo pipefail
fresh_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$fresh_repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc" HF_LEROBOT_HOME="$HF_HOME/lerobot"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DISABLE_XET=1
export DVLA_WANDB_ENTITY=aparame-clemson-university CUDA_VISIBLE_DEVICES=0,1
export SLURM_MEM_PER_NODE=231673
exec "$KRVLA_WORK/envs/train/bin/python" -u \
  "$KRVLA_WORK/training/fresh20k-20261025/controller/fresh20k_training.py" run \
  --source "$KRVLA_WORK/training/krv100-full-20261010/source" \
  --stage "$KRVLA_WORK/training/fresh20k-20261025"
