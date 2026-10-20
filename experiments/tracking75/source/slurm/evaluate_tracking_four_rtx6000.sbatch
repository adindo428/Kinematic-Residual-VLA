#!/usr/bin/env bash
#SBATCH --job-name=tracking-four-rtx6000
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err
set -euo pipefail
cd /home/aparame/Research/Dynamic_VLA/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA
export EVAL_STAGE=all
export VELOCITY=${VELOCITY:-/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_velocity/checkpoints/tracking_dom75_full20_velocity_16680047}
# The shared driver is executed as Bash; its SBATCH directives do not apply.
exec bash slurm/evaluate_tracking_four_l40.sbatch
