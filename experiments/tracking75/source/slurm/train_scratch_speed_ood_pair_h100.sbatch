#!/usr/bin/env bash
#SBATCH --job-name=dvla-scratch-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err

set -euo pipefail

cd /home/aparame/Research/Dynamic_VLA/third_party/DynamicVLA

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-4}
export DVLA_N_WORKERS=${DVLA_N_WORKERS:-4}
export DVLA_DATALOADER_PREFETCH_FACTOR=${DVLA_DATALOADER_PREFETCH_FACTOR:-2}
export DVLA_PERSISTENT_WORKERS=${DVLA_PERSISTENT_WORKERS:-true}

export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export AUTO_RESUME=${AUTO_RESUME:-1}
export DVLA_BATCH_SIZE=${DVLA_BATCH_SIZE:-256}
export DVLA_GRAD_ACCUM_STEPS=${DVLA_GRAD_ACCUM_STEPS:-2}
bash scripts/train_scratch_speed_ood_pair.sh
