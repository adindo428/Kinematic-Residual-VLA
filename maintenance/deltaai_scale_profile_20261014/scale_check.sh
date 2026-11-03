#!/usr/bin/env bash
set -euo pipefail
source /work/nvme/bivv/aparameshwaran/dynamicvla/setup-20261008/activate-training.sh
export DVLA_PROJECT_DIR=/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100
export DOM100_ROOT=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
cd "$DVLA_PROJECT_DIR"
timeout --signal=TERM --kill-after=30s 13m python /work/nvme/bivv/aparameshwaran/dynamicvla/setup-scale-domv-20261014/scale_check.py
