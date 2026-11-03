#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DVLA_PROJECT_DIR="${PROJECT_DIR}"
export DOM100_ROOT=${DOM100_ROOT:-/scratch/aparame/Dynamic_VLA/dom100}
gpu=${DOM100_GPU:-a100}
case "${gpu}" in a100|h100|h200) ;; *) echo "Unsupported DOM100_GPU: ${gpu}" >&2; exit 2 ;; esac
cd "${PROJECT_DIR}"
source scripts/dom100_env.sh
# Login-node validation does not initialize CUDA or construct the model.
python scripts/dom100_submit.py --family-root "${DOM100_ROOT}" --gpu "${gpu}" "$@"

