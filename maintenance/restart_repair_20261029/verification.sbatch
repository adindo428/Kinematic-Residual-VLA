#!/usr/bin/env bash
#SBATCH --job-name=nvlfix-cpu-verify
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/restart-repair-20261029-r2/logs/verify-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/restart-repair-20261029-r2/logs/verify-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /scratch/aparame/Dynamic_VLA/restart-repair-20261029-r2
/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python verify.py
