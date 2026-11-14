#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$KRVLA_WORK/envs/eval20k/bin/python" -u "$repo/workstation/evaluate_smolvla_after_krv100.py" "$@"
