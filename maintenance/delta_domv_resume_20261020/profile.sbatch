#!/usr/bin/env bash
#SBATCH --account=bivv-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=00:30:00
#SBATCH --job-name=domv-h200-profile
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-delta-domv-resume-20261020/logs/profile-%j.out
#SBATCH --error=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-delta-domv-resume-20261020/logs/profile-%j.err
set -euo pipefail
HERE=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-delta-domv-resume-20261020
source "$HERE/activate.sh"
python "$HERE/prepare.py"
export DVLA_EPOCH_CONTROL_FILE="$HERE/control.json"
export WANDB_MODE=disabled
timeout --signal=TERM --kill-after=30s 26m python "$HERE/profile.py"
