#!/usr/bin/env bash
set -euo pipefail
finish_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$finish_repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc" HF_LEROBOT_HOME="$HF_HOME/lerobot"
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export DVLA_WANDB_ENTITY=aparame-clemson-university
finish_stage="$KRVLA_WORK/training/krv100-20k-from-epoch003-20261021"
exec "$KRVLA_WORK/envs/train/bin/python" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$finish_stage/continuations/finish-epoch10-20261024/finish_krv100_epoch10.py" \
  --stage "$finish_stage"
