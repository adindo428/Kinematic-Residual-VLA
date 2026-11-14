#!/usr/bin/env bash
set -euo pipefail
continuation_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$continuation_repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export DVLA_WANDB_ENTITY=aparame-clemson-university
continuation_stage="$KRVLA_WORK/training/krv100-20k-from-epoch003-20261021"
exec "$KRVLA_WORK/envs/train/bin/python" -u \
  "$continuation_stage/controller/continue_krv100_20k.py" wait \
  --source "$KRVLA_WORK/training/krv100-full-20261010/source" \
  --stage "$continuation_stage"
