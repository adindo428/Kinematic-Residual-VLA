#!/usr/bin/env bash
#SBATCH --job-name=dvla-ff-a100-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err

set -euo pipefail

cd /home/aparame/Research/Dynamic_VLA/third_party/DynamicVLA

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-4}
export DVLA_N_WORKERS=${DVLA_N_WORKERS:-2}
export DVLA_DATALOADER_PREFETCH_FACTOR=${DVLA_DATALOADER_PREFETCH_FACTOR:-2}
export DVLA_PERSISTENT_WORKERS=${DVLA_PERSISTENT_WORKERS:-true}

export NPROC_PER_NODE=4
export AUTO_RESUME=1
export DVLA_BATCH_SIZE=${DVLA_BATCH_SIZE:-8}
export DVLA_GRAD_ACCUM_STEPS=${DVLA_GRAD_ACCUM_STEPS:-32}
export RUN_PREFETCH=0
export RUN_TRAIN=1
export RUN_OFFLINE_COMPARE=0
export DVLA_OUTPUT_ROOT=/scratch/aparame/Dynamic_VLA/runs/a100_4gpu
export BASELINE_EXP=dynamicvla_dom_full_finetune_speed_ood_0p10_0p18_a100_4gpu
export RESIDUAL_EXP=residual_dynamicvla_dom_full_finetune_speed_ood_0p10_0p18_a100_4gpu

attempt=1
max_attempts=3
while (( attempt <= max_attempts )); do
  echo "[full-finetune-speed-ood] A100 attempt ${attempt}/${max_attempts}"
  if bash scripts/train_full_finetune_speed_ood_pair.sh; then
    exit 0
  fi
  if (( attempt == max_attempts )); then
    echo "[full-finetune-speed-ood] Training failed after ${max_attempts} attempts." >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  sleep 30
done
