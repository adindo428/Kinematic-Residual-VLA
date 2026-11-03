#!/usr/bin/env bash
set -euo pipefail
source /work/nvme/bivv/aparameshwaran/dynamicvla/setup-20261008/activate-training.sh
export DVLA_PROJECT_DIR=/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100
export DOM100_ROOT=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-stress-ddp-20261014
export DOM100_DDP_GPUS=4 GLOO_SOCKET_IFNAME=lo
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
cd "$DVLA_PROJECT_DIR"
timeout --signal=TERM --kill-after=30s 37m python /work/nvme/bivv/aparameshwaran/dynamicvla/setup-stress-domv-20261014/stress_check.py >"$DOM100_ROOT/logs/stress-check-$SLURM_JOB_ID.log" 2>&1
