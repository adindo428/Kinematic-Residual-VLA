#!/usr/bin/env bash
set -euo pipefail
: "${DVLA_PROJECT_DIR:?Set the frozen DeltaAI training project}"
: "${DOM100_ROOT:?Set the frozen DeltaAI training family}"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=4 PYTHONUNBUFFERED=1
