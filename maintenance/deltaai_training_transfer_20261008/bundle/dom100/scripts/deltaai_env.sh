#!/usr/bin/env bash
set -euo pipefail

: "${DVLA_ACCESS_PROJECT:?Set DVLA_ACCESS_PROJECT to your DeltaAI project serial code}"
export DVLA_WORK_ROOT=${DVLA_WORK_ROOT:-/work/nvme/${DVLA_ACCESS_PROJECT}/${USER}/dynamicvla}
export DVLA_OUTPUT_ROOT=${DVLA_OUTPUT_ROOT:-${DVLA_WORK_ROOT}/runs}
export DVLA_LOG_ROOT=${DVLA_LOG_ROOT:-${DVLA_WORK_ROOT}/logs}
export HF_HOME=${HF_HOME:-${DVLA_WORK_ROOT}/huggingface}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}
export WANDB_DIR=${WANDB_DIR:-${DVLA_LOG_ROOT}/wandb}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-${DVLA_WORK_ROOT}/wandb-config}
export TORCH_HOME=${TORCH_HOME:-${HF_HOME}/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${DVLA_WORK_ROOT}/cache}
mkdir -p "${DVLA_OUTPUT_ROOT}" "${DVLA_LOG_ROOT}" "${HF_LEROBOT_HOME}" "${WANDB_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}"

: "${DVLA_PYTHON_ENV:?Set DVLA_PYTHON_ENV to a tested DeltaAI ARM Python environment}"
module load "${DVLA_PYTORCH_MODULE:-python/miniforge3_pytorch/2.12.0}"
source "${DVLA_PYTHON_ENV}/bin/activate"

if [[ -n "${DVLA_SECRETS_FILE:-}" ]]; then
  [[ -f "${DVLA_SECRETS_FILE}" && ! -L "${DVLA_SECRETS_FILE}" ]] || {
    echo "Missing or symlinked DVLA_SECRETS_FILE" >&2
    return 1
  }
  permissions=$(stat -c %a "${DVLA_SECRETS_FILE}")
  [[ "${permissions}" == "600" || "${permissions}" == "400" ]] || {
    echo "DVLA_SECRETS_FILE must have permissions 600 or 400" >&2
    return 1
  }
  set -a
  source "${DVLA_SECRETS_FILE}"
  set +a
fi

[[ -n "${WANDB_API_KEY:-}" ]] || {
  echo "This project requires WANDB_API_KEY in DVLA_SECRETS_FILE; refusing to use the shared ~/.netrc login." >&2
  return 1
}
[[ -n "${HF_TOKEN:-}" ]] || {
  echo "This project requires HF_TOKEN in DVLA_SECRETS_FILE." >&2
  return 1
}
