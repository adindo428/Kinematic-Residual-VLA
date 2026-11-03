#!/bin/bash -l
#SBATCH --job-name=dom-clean-pack
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/clean-dom-four-gpu-20261012/pack-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/clean-dom-four-gpu-20261012/pack-%j.err
set -euo pipefail
python3 /scratch/aparame/Dynamic_VLA/clean-dom-four-gpu-20261012/pack.py /scratch/aparame/Dynamic_VLA/dom100-audited-20261012 /scratch/aparame/Dynamic_VLA/clean-dom-four-gpu-20261012/dom-clean.tar
