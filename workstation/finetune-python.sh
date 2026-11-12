#!/usr/bin/env bash
# Isolated native KRV100/DOMV training Python; no robot hardware is opened.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ENV="$PROJECT_ROOT/.venv-finetune"
if [[ ! -x "$TRAIN_ENV/bin/python" || ! -d "$TRAIN_ENV/ffmpeg/lib" ]]; then
    echo "Missing isolated fine-tuning Python or FFmpeg runtime: $TRAIN_ENV" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$TRAIN_ENV/ffmpeg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The collection Conda interpreter has an older C++ runtime in its RPATH.
# Load the compatible isolated runtime before importing Torch/FFmpeg.
export LD_PRELOAD="$TRAIN_ENV/ffmpeg/lib/libstdc++.so.6:$TRAIN_ENV/ffmpeg/lib/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
exec "$TRAIN_ENV/bin/python" "$@"
