#!/usr/bin/env bash
#SBATCH --account=bivv-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=07:00:00
#SBATCH --job-name=domv-resume-h200
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-delta-domv-resume-20261020/logs/train-%j.out
#SBATCH --error=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-delta-domv-resume-20261020/logs/train-%j.err
set -euo pipefail
HERE=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-delta-domv-resume-20261020
source "$HERE/activate.sh"
python "$HERE/prepare.py"
export DVLA_EPOCH_CONTROL_FILE="$HERE/control.json"
export WANDB_MODE=disabled
timeout --signal=TERM --kill-after=30s 16m python "$HERE/check.py"
export WANDB_MODE=online
exec bash "$HERE/launch_ranks.sh" train
