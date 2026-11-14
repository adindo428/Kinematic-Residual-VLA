#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -f "$KRVLA_WORK/evaluations/low-clutter-full-table1-20261102/krv_full_two_suites/enabled" ]]; then
  exec "$KRVLA_WORK/envs/eval20k/bin/python" -u "$repo/workstation/run_krv_full_two_suites.py" "$@"
fi
exec "$KRVLA_WORK/envs/eval20k/bin/python" -u "$repo/workstation/run_parallel_low_clutter_table1.py" "$@"
