#!/usr/bin/env bash
#SBATCH --job-name=tracking-velocity-smoke
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err
set -euo pipefail
PROJECT=/home/aparame/Research/Dynamic_VLA/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA
cd "$PROJECT"
export HF_HOME=/scratch/aparame/.cache/huggingface
export HF_LEROBOT_HOME=$HF_HOME/lerobot
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/software/spackages/linux-rocky8-x86_64/gcc-9.5.0/cuda-12.1.1-d4ik3tbvtk3ypske36p4ogeulsnsv26p/lib64:${LD_LIBRARY_PATH:-}
unset LEROBOT_HOME
PYTHON=/home/aparame/.conda/envs/dynamicvla/bin/python
"$PYTHON" -u scripts/smoke_tracking_velocity.py --config configs/dynamicvla_dom_tracking_velocity75.yaml \
 --output "/home/aparame/Research/Dynamic_VLA/RESULTS/tracking_velocity_verification/$SLURM_JOB_ID"
"$PYTHON" -u scripts/verify_tracking_training.py --variant residual \
 --checkpoint /scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/checkpoints/tracking_dom75_full20_residual_16623337
