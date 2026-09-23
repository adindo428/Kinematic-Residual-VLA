#!/usr/bin/env bash
#SBATCH --job-name=dvla-paired-vel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --output=runs/slurm_logs/%x-%j.out
#SBATCH --error=runs/slurm_logs/%x-%j.err

set -euo pipefail

cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
mkdir -p runs/slurm_logs runs/proactive_vla/paired_velocity_sweep

if command -v module >/dev/null 2>&1; then
  module load cuda/12.3.0 || true
  module load gcc/12.3.0 || true
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate dynamicvla
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CFG=${CFG:-configs/residual_dynamicvla_dom_paired_velocity.yaml}
BASELINE_CKPT=${BASELINE_CKPT:-runs/checkpoints/dynamicvla_dom_paired_velocity}
RESIDUAL_CKPT=${RESIDUAL_CKPT:-runs/checkpoints/residual_dynamicvla_dom_paired_velocity}
OUTPUT=${OUTPUT:-runs/proactive_vla/paired_velocity_sweep/report.json}
SPLIT=${SPLIT:-train}
MAX_BATCHES=${MAX_BATCHES:-500}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}

python scripts/evaluate_residual_paired_velocity_sweep.py \
  --cfg "${CFG}" \
  --baseline-checkpoint "${BASELINE_CKPT}" \
  --residual-checkpoint "${RESIDUAL_CKPT}" \
  --output "${OUTPUT}" \
  --split "${SPLIT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-batches "${MAX_BATCHES}" \
  --cached-only \
  --shuffle-episodes \
  --counterfactual-speeds 0.02 0.08 0.12 0.18 0.25 0.35 \
  --residual-ablations correct zero wrong_sign shuffle \
  --progress-every 25
