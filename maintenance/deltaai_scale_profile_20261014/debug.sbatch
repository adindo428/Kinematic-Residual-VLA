#!/usr/bin/env bash
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=00:20:00
#SBATCH --exclude=gh055
#SBATCH --job-name=domv-scale8-check
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014/logs/debug-%j.out
#SBATCH --error=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014/logs/debug-%j.err
set -euo pipefail
HERE=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-scale-domv-20261014
source /work/nvme/bivv/aparameshwaran/dynamicvla/setup-20261008/activate-training.sh
export DVLA_PROJECT_DIR=/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100
export DOM100_ROOT=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
cd "$DVLA_PROJECT_DIR"
timeout --signal=TERM --kill-after=30s 13m python "$HERE/scale_check.py"
python "$HERE/approve_runtime.py"
python "$HERE/launch.py"
# Allocation exits here; full training is a separate regular48hour job with no dependency.
