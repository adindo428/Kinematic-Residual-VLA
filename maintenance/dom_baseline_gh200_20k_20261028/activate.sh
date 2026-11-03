#!/usr/bin/env bash
set -euo pipefail
export DOM_BASELINE_ROOT=/work/nvme/bicn/anayak2/dynamicvla/dom100-20k-gh200-20261028
module load python/miniforge3_pytorch/2.12.0
export VIRTUAL_ENV="$DOM_BASELINE_ROOT/environment"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME
export DVLA_PROJECT_DIR="$DOM_BASELINE_ROOT/source"
export DOM100_ROOT="$DOM_BASELINE_ROOT/family"
export DOM100_RUNTIME_FILE="$DOM100_ROOT/selected-runtime.json"
export PYTHONPATH="$DOM100_ROOT:$DVLA_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$DOM_BASELINE_ROOT/native/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$DOM_BASELINE_ROOT/huggingface"
export HF_HUB_CACHE="$DOM_BASELINE_ROOT/tokenizer-cache"
export HF_LEROBOT_HOME=/work/nvme/bicn/anayak2/dynamicvla/huggingface/lerobot
export XDG_CACHE_HOME="$DOM_BASELINE_ROOT/cache"
export TORCH_HOME="$DOM_BASELINE_ROOT/cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export WANDB_MODE=offline WANDB_DIR="$DOM_BASELINE_ROOT/logs/wandb"
export WANDB_CONFIG_DIR="$DOM_BASELINE_ROOT/wandb-config"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MALLOC_ARENA_MAX=4
mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$TORCH_HOME"
