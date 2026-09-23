#!/usr/bin/env bash
#SBATCH --job-name=dvla-ff-iid-ood-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:1
#SBATCH --time=72:00:00
#SBATCH --mem=256GB
#SBATCH --output=runs/slurm_logs/%x-%j.out
#SBATCH --error=runs/slurm_logs/%x-%j.err

set -euo pipefail

cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
mkdir -p runs/slurm_logs runs/proactive_vla/offline_compare

if command -v module >/dev/null 2>&1; then
  module load cuda/12.3.0 || true
  module load gcc/12.3.0 || true
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate dynamicvla
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-4}
export DVLA_N_WORKERS=${DVLA_N_WORKERS:-4}

EVAL_CFG=${EVAL_CFG:-configs/residual_dynamicvla_dom_full_finetune_speed_ood_0p12_0p13.yaml}
BASELINE_CKPT=${BASELINE_CKPT:-runs/checkpoints/dynamicvla_dom_full_finetune_speed_ood_0p12_0p13}
RESIDUAL_CKPT=${RESIDUAL_CKPT:-runs/checkpoints/residual_dynamicvla_dom_full_finetune_speed_ood_0p12_0p13}
EVAL_OUTPUT=${EVAL_OUTPUT:-runs/proactive_vla/offline_compare/full_finetune_iid_ood_report.json}
EVAL_SPLITS=${EVAL_SPLITS:-iid ood}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
# Set MAX_BATCHES here, or override at submit time:
#   MAX_BATCHES=500 sbatch slurm/evaluate_full_finetune_iid_ood_h100.sbatch
# With the default EVAL_BATCH_SIZE=1, MAX_BATCHES=20000 scores 20000 batches/samples per split.
MAX_BATCHES=${MAX_BATCHES:-2000}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}
# Keep evaluation on already-cached complete episodes by default.
# Set EVAL_CACHED_ONLY=0 to restore the old download-missing-files behavior.
EVAL_CACHED_ONLY=${EVAL_CACHED_ONLY:-1}
# Optional cap on cached episodes selected per split before dataset construction; 0 keeps all cached episodes.
MAX_CACHED_EPISODES=${MAX_CACHED_EPISODES:-0}
read -r -a EVAL_SPLIT_ARGS <<< "${EVAL_SPLITS}"
EVAL_CACHE_ARGS=()
if [[ "${EVAL_CACHED_ONLY}" == "1" ]]; then
  EVAL_CACHE_ARGS+=(--cached-only)
fi
if [[ "${MAX_CACHED_EPISODES}" != "0" ]]; then
  EVAL_CACHE_ARGS+=(--max-cached-episodes "${MAX_CACHED_EPISODES}")
fi

python scripts/evaluate_baseline_residual_iid_ood.py \
  --cfg "${EVAL_CFG}" \
  --baseline-checkpoint "${BASELINE_CKPT}" \
  --residual-checkpoint "${RESIDUAL_CKPT}" \
  --output "${EVAL_OUTPUT}" \
  --splits "${EVAL_SPLIT_ARGS[@]}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --max-batches "${MAX_BATCHES}" \
  "${EVAL_CACHE_ARGS[@]}" \
  --progress-every "${PROGRESS_EVERY}"
