#!/bin/bash -l
#SBATCH --job-name=vlaopt-resume
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-opt-20261019/logs/prepare-resume-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-opt-20261019/logs/prepare-resume-%j.err
set -euo pipefail
cd /scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-opt-20261019
export LD_LIBRARY_PATH=/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2
env/bin/python prepare_resume.py
