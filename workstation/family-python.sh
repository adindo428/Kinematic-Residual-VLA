#!/usr/bin/env bash
set -euo pipefail
FAMILY_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FAMILY_ROOT="${POSTTRAIN_ROOT:-$FAMILY_PROJECT/scratch/real_robot/family-finetuning-v6}"
export PYTHONPATH="$FAMILY_ROOT/dependencies:$FAMILY_ROOT/hub/code/runtime-adapters:$FAMILY_PROJECT/third_party/DynamicVLA:$FAMILY_PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ -n "${POSTTRAIN_PYTHON:-}" ]]; then
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
    if [[ -n "${POSTTRAIN_FFMPEG_LIB:-}" ]]; then
        export LD_LIBRARY_PATH="$POSTTRAIN_FFMPEG_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    exec "$POSTTRAIN_PYTHON" "$@"
fi
exec "$FAMILY_PROJECT/workstation/finetune-python.sh" "$@"
