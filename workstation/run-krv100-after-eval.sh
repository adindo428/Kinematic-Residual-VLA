#!/usr/bin/env bash
set -euo pipefail
training_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$training_repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
exec "$POLICY_PYTHON" -u "$training_repo/workstation/prepare_krv100_training.py"
