# Source from any directory; this selects the relocated runtime for child processes.
POSTTRAIN_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export POSTTRAIN_ROOT="${POSTTRAIN_ROOT:-$POSTTRAIN_PROJECT/scratch/real_robot/family-finetuning-v7}"
export POSTTRAIN_PYTHON="${POSTTRAIN_PYTHON:-$POSTTRAIN_PROJECT/.venv-post-training/bin/python}"
export PYTHONPATH="$POSTTRAIN_ROOT/hub/code/runtime-adapters:$POSTTRAIN_PROJECT/third_party/DynamicVLA:$POSTTRAIN_PROJECT/workstation:$POSTTRAIN_PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
if [[ -n "${POSTTRAIN_FFMPEG_LIB:-}" ]]; then
    export LD_LIBRARY_PATH="$POSTTRAIN_FFMPEG_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Preparation downloads are online; the actual model wrapper uses offline assets.
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
