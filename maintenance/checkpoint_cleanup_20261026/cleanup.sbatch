#!/usr/bin/env bash
#SBATCH --job-name=checkpoint-cleanup-oct26
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/checkpoint-cleanup-20261026/cleanup-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/checkpoint-cleanup-20261026/cleanup-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /scratch/aparame/Dynamic_VLA/checkpoint-cleanup-20261026
/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python -u cleanup.py
