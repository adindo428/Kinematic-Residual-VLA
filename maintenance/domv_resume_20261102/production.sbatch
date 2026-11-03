#!/usr/bin/env bash
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --job-name=domv-resume-2n
#SBATCH --time=48:00:00
#SBATCH --exclude=gh055
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-domv-resume-20261102/logs/train-%j.out
#SBATCH --error=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-domv-resume-20261102/logs/train-%j.err
set -euo pipefail
module load python/miniforge3_pytorch/2.12.0
export VIRTUAL_ENV=/work/nvme/bivv/aparameshwaran/dynamicvla/venv-krvla-dom-20261008
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME
export DVLA_PROJECT_DIR=/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100
export DOM100_ROOT=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014
export DOM100_RUNTIME_FILE="$DOM100_ROOT/selected-runtime.json"
export DOMV_LAUNCH=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-domv-resume-20261102
export DVLA_EPOCH_CONTROL_FILE="$DOMV_LAUNCH/control.json"
export PYTHONPATH="$DOMV_LAUNCH:$DVLA_PROJECT_DIR"
export LD_LIBRARY_PATH="/work/nvme/bivv/aparameshwaran/dynamicvla/native-krvla-dom/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/work/nvme/bivv/aparameshwaran/dynamicvla/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_LEROBOT_HOME="$HF_HOME/lerobot" TORCH_HOME="$HF_HOME/torch"
export XDG_CACHE_HOME=/work/nvme/bivv/aparameshwaran/dynamicvla/cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export WANDB_CONFIG_DIR="$DOMV_LAUNCH/wandb-config"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MALLOC_ARENA_MAX=4
cd "$DVLA_PROJECT_DIR"
python "$DOMV_LAUNCH/preflight.py"
exec bash "$DOM100_ROOT/run_nodes.sh" train
