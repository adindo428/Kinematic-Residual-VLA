#!/usr/bin/env bash
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --job-name=domv-2n-gh200
#SBATCH --time=48:00:00
#SBATCH --exclude=gh055
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014/logs/train-%j.out
#SBATCH --error=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014/logs/train-%j.err
set -euo pipefail
source /work/nvme/bivv/aparameshwaran/dynamicvla/setup-20261008/activate-training.sh
export DVLA_PROJECT_DIR=/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100
export DOM100_ROOT=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014
export DOM100_RUNTIME_FILE="$DOM100_ROOT/selected-runtime.json"
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
HERE=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-scale-domv-20261014
cd "$DVLA_PROJECT_DIR"
python "$HERE/prepare_production.py"
export WANDB_MODE=online
exec bash "$DOM100_ROOT/run_nodes.sh" train
