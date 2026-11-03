#!/usr/bin/env bash
set -euo pipefail
module load pytorch-conda/2.12
export DVLA_WORK_ROOT=/work/nvme/bivv/aparameshwaran/dynamicvla
source "$DVLA_WORK_ROOT/venv-domv-delta-20261020/bin/activate"
export DVLA_PROJECT_DIR="$DVLA_WORK_ROOT/training-sources/delta-domv-f991015a"
export DOM100_ROOT="$DVLA_WORK_ROOT/domv-scale-ddp-20261014"
export HF_HOME="$DVLA_WORK_ROOT/huggingface" HF_LEROBOT_HOME="$DVLA_WORK_ROOT/huggingface/lerobot"
export HF_HUB_CACHE="$HF_HOME/hub" TORCH_HOME="$HF_HOME/torch" XDG_CACHE_HOME="$DVLA_WORK_ROOT/cache"
export WANDB_CONFIG_DIR="$DVLA_WORK_ROOT/wandb-config" WANDB_DIR="$DOM100_ROOT/runs/domv100"
secret=/u/aparameshwaran/.config/kinematic-residual-vla/secrets-delta.env
[[ -f "$secret" && ! -L "$secret" && $(stat -c %a "$secret") == 600 ]]
set -a
source "$secret"
set +a
: "${WANDB_API_KEY:?Missing project W&B credentials}"
export PYTHONPATH="$DVLA_WORK_ROOT/setup-delta-domv-resume-20261020:$DVLA_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export GLOO_SOCKET_IFNAME=lo NCCL_DEBUG=WARN
unset DVLA_EPOCH_CONTROL_FILE DOM100_NODE_MODE
cd "$DVLA_PROJECT_DIR"
