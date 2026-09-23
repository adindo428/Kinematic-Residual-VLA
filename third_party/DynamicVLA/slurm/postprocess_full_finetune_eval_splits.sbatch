#!/usr/bin/env bash
#SBATCH --job-name=dvla-ff-eval-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --mem=96GB
#SBATCH --output=runs/slurm_logs/%x-%j.out
#SBATCH --error=runs/slurm_logs/%x-%j.err

set -euo pipefail

cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
mkdir -p runs/slurm_logs

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
export DYNAMICVLA_PREFETCH_STATUS_INTERVAL=${DYNAMICVLA_PREFETCH_STATUS_INTERVAL:-300}
export DYNAMICVLA_PREFETCH_RETRIES=${DYNAMICVLA_PREFETCH_RETRIES:-5}
export DYNAMICVLA_PREFETCH_CHUNK_PATTERN_THRESHOLD=${DYNAMICVLA_PREFETCH_CHUNK_PATTERN_THRESHOLD:-1000}

EVAL_CFG=${EVAL_CFG:-configs/residual_dynamicvla_dom_full_finetune_speed_ood_0p12_0p13.yaml}
EVAL_SPLITS=${EVAL_SPLITS:-ood}
read -r -a EVAL_SPLIT_ARGS <<< "${EVAL_SPLITS}"

python scripts/prefetch_eval_splits.py \
  --cfg "${EVAL_CFG}" \
  --splits "${EVAL_SPLIT_ARGS[@]}"
