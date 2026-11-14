#!/usr/bin/env bash
set -euo pipefail
resume_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$resume_repo/workstation/env.local.sh"
export NETRC="$KRVLA_WORK/config/wandb/netrc"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset WANDB_SERVICE
exec "$KRVLA_WORK/envs/train/bin/python" -u "$resume_repo/workstation/resume_krv100_after_eval.py" \
  --request "$KRVLA_WORK/training/krv100-full-20261010/post_eval_resume_request.json"
