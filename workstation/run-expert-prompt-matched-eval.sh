#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
root="$KRVLA_WORK/evaluations/expert-prompt-matched20k-20261102"
if [[ ! -f "$root/expert_scene_preparation_complete.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$ISAACLAB" -p "$repo/workstation/prepare_expert_prompt_matched_eval.py" --generate
fi
exec "$KRVLA_WORK/envs/eval20k/bin/python" -u "$repo/workstation/run_expert_prompt_matched_eval.py" "$@"
