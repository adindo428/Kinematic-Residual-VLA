#!/bin/bash -l
#SBATCH --job-name=pi05-ready-release
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-release-20261020/logs/pi05_baseline-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-release-20261020/logs/pi05_baseline-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2
export LD_LIBRARY_PATH=/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-pi05-cache-20261020/env/bin/python /scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-release-20261020/verify_ready.py /scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-release-20261020/pi05_baseline-config.json
