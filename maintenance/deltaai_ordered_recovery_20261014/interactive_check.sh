#!/usr/bin/env bash
set -euo pipefail
source /work/nvme/bivv/aparameshwaran/dynamicvla/setup-20261008/activate-training.sh
export DVLA_PROJECT_DIR=/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100
export DOM100_ROOT=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-ordered-ddp-20261014
export DOM100_DDP_GPUS=4 GLOO_SOCKET_IFNAME=lo
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export DOM100_QUEUED_TRAINING_JOB=3379435
cd "$DVLA_PROJECT_DIR"
python - <<'PY_HARDWARE'
import os,json,torch
assert torch.cuda.device_count()==4
print(json.dumps({'interactive_job':os.environ['SLURM_JOB_ID'],'node':os.environ['SLURM_NODELIST'],'GPUs':[torch.cuda.get_device_name(i) for i in range(4)]}),flush=True)
PY_HARDWARE
HERE=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-ordered-domv-20261014
timeout --signal=TERM --kill-after=30s 16m python "$HERE/interactive_check.py" >"$DOM100_ROOT/logs/interactive-check-$SLURM_JOB_ID.log" 2>&1
cat "$HERE/interactive-check-$SLURM_JOB_ID.json"
