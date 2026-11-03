#!/usr/bin/env bash
#SBATCH --job-name=dvla-paired-vel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err

set -euo pipefail

cd /home/aparame/Research/Dynamic_VLA/third_party/DynamicVLA
source scripts/palmetto_env.sh
mkdir -p "${DVLA_OUTPUT_ROOT}/proactive_vla/paired_velocity_sweep"

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CFG=${CFG:-configs/residual_dynamicvla_dom_paired_velocity.yaml}
BASELINE_CKPT=${BASELINE_CKPT:-${DVLA_OUTPUT_ROOT}/checkpoints/dynamicvla_dom_paired_velocity}
RESIDUAL_CKPT=${RESIDUAL_CKPT:-${DVLA_OUTPUT_ROOT}/checkpoints/residual_dynamicvla_dom_paired_velocity}
OUTPUT=${OUTPUT:-${DVLA_OUTPUT_ROOT}/proactive_vla/paired_velocity_sweep/report.json}
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
