#!/usr/bin/env bash
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=domv-wandb-live
#SBATCH --output=/scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114/logs/wandb-sync-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114/logs/wandb-sync-%j.err
set -euo pipefail
source /scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114/env.sh
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=''
export GOMAXPROCS=1
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
exec srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
  wandb beta sync --live --no-skip-synced \
  --entity aparame-clemson-university --project Kinematic-Residual-VLA --id bs8kmkz1 \
  /scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114/wandb/offline-run-20261114_095417-bs8kmkz1
