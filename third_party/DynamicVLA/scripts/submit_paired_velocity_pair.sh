#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

if (( $# > 0 )); then
  GPU_TYPE=$1
  shift
else
  GPU_TYPE=${GPU_TYPE:-h100}
fi

if (( $# > 0 )); then
  GPU_COUNT=$1
  shift
else
  GPU_COUNT=${GPU_COUNT:-8}
fi

GPU_TYPE=${GPU_TYPE,,}
case "${GPU_TYPE}" in
  a100|h100|h200) ;;
  *)
    echo "GPU type must be one of: a100, h100, h200. Got: ${GPU_TYPE}" >&2
    exit 2
    ;;
esac

if [[ ! "${GPU_COUNT}" =~ ^[0-9]+$ ]] || (( GPU_COUNT < 4 || GPU_COUNT > 8 )); then
  echo "GPU count must be an integer from 4 through 8. Got: ${GPU_COUNT}" >&2
  exit 2
fi

export DVLA_REQUESTED_GPU_TYPE=${GPU_TYPE}
export DVLA_REQUESTED_GPU_COUNT=${GPU_COUNT}

cd "${PROJECT_ROOT}"
mkdir -p runs/slurm_logs

echo "[paired-velocity] Requesting ${GPU_COUNT} x ${GPU_TYPE}."
exec sbatch \
  --gres="gpu:${GPU_TYPE}:${GPU_COUNT}" \
  "$@" \
  slurm/train_paired_velocity_pair_h100.sbatch
