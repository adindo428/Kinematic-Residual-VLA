#!/usr/bin/env bash
#SBATCH --job-name=dvla-ff-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err

set -euo pipefail

cd /home/aparame/Research/Dynamic_VLA/third_party/DynamicVLA

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-4}
export DYNAMICVLA_PREFETCH_STATUS_INTERVAL=${DYNAMICVLA_PREFETCH_STATUS_INTERVAL:-300}
export DYNAMICVLA_PREFETCH_RETRIES=${DYNAMICVLA_PREFETCH_RETRIES:-5}
export DYNAMICVLA_PREFETCH_CHUNK_PATTERN_THRESHOLD=${DYNAMICVLA_PREFETCH_CHUNK_PATTERN_THRESHOLD:-1000}
export RUN_PREFETCH=${RUN_PREFETCH:-1}
export RUN_TRAIN=${RUN_TRAIN:-0}
export RUN_OFFLINE_COMPARE=${RUN_OFFLINE_COMPARE:-0}

bash scripts/train_full_finetune_speed_ood_pair.sh
