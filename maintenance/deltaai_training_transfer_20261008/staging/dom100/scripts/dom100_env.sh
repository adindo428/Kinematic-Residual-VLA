#!/usr/bin/env bash
set -euo pipefail
export DVLA_PROJECT_DIR=${DVLA_PROJECT_DIR:-$PWD}
export DOM100_ROOT=${DOM100_ROOT:-/scratch/aparame/Dynamic_VLA/dom100}
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=4
export PYTHONUNBUFFERED=1
source "${DVLA_PROJECT_DIR}/scripts/palmetto_env.sh"
if [[ ! -f "${DVLA_PROJECT_DIR}/scripts/dom100.py" ]]; then
    echo "DVLA_PROJECT_DIR must select the canonical checkout containing scripts/dom100.py" >&2
    exit 2
fi

