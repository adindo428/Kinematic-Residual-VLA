#!/bin/bash
# Source this file inside a DeltaAI compute allocation.
export DVLA_ACCESS_PROJECT=bivv
export DVLA_WORK_ROOT=/work/nvme/bivv/aparameshwaran/dynamicvla
export DVLA_PYTHON_ENV=$DVLA_WORK_ROOT/venv-krvla-dom-20261008
export DVLA_PYTORCH_MODULE=python/miniforge3_pytorch/2.12.0
export DVLA_SECRETS_FILE=/u/aparameshwaran/.config/kinematic-residual-vla/secrets.env
export DVLA_SNAPSHOT_ROOT=/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261008T201101Z
export DVLA_PROJECT_DIR=$DVLA_SNAPSHOT_ROOT/dom100
export DVLA_ENV_SCRIPT=$DVLA_WORK_ROOT/setup-20261008/activate-training.sh
export DOM100_ROOT=$DVLA_WORK_ROOT/dom100-palmetto-h20
export HF_HOME=$DVLA_WORK_ROOT/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_LEROBOT_HOME=$HF_HOME/lerobot
export TORCH_HOME=$HF_HOME/torch
export XDG_CACHE_HOME=$DVLA_WORK_ROOT/cache
export DVLA_OUTPUT_ROOT=$DVLA_WORK_ROOT/runs
export DVLA_LOG_ROOT=$DVLA_WORK_ROOT/logs
export WANDB_DIR=$DVLA_LOG_ROOT/wandb
export WANDB_CONFIG_DIR=$DVLA_WORK_ROOT/wandb-config
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=4 PYTHONUNBUFFERED=1
source "$DVLA_PROJECT_DIR/scripts/deltaai_env.sh"
export LD_LIBRARY_PATH="$DVLA_WORK_ROOT/native-krvla-dom/lib:${LD_LIBRARY_PATH:-}"
