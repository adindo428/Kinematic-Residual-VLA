#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-/scratch/aparame/.cache/huggingface}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}
export DVLA_OUTPUT_ROOT=${DVLA_OUTPUT_ROOT:-/scratch/aparame/Dynamic_VLA/runs}
export DVLA_LOG_ROOT=${DVLA_LOG_ROOT:-/home/aparame/Research/Dynamic_VLA/logs}
export WANDB_DIR=${WANDB_DIR:-${DVLA_LOG_ROOT}/wandb}
export TORCH_HOME=${TORCH_HOME:-${HF_HOME}/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/scratch/aparame/.cache}
mkdir -p "${DVLA_OUTPUT_ROOT}" "${DVLA_LOG_ROOT}/slurm" "${WANDB_DIR}"

if command -v module >/dev/null 2>&1; then
  module load gcc/12.3.0
  module load cuda/12.3.0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dynamicvla
