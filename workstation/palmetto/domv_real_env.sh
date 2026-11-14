#!/usr/bin/env bash
set -euo pipefail
export DOMV_ROOT=/scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114
export PATH="/home/aparame/.conda/envs/dynamicvla/bin:$PATH"
export LD_LIBRARY_PATH="/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$DOMV_ROOT/runtime-deps:$DOMV_ROOT/app/workstation:$DOMV_ROOT/app/third_party/DynamicVLA:$DOMV_ROOT/app${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/aparame/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_DIR="$DOMV_ROOT"
export WANDB_MODE=offline
