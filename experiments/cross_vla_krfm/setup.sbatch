#!/bin/bash -l
#SBATCH --job-name=kr-backbones-setup
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/logs/setup-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/logs/setup-%j.err
set -euo pipefail
cd /scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019
export HF_HOME=/scratch/aparame/.cache/huggingface
export PYTHONPATH=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/upstream-vlash
export LD_LIBRARY_PATH=/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
env/bin/python -m unittest -v test_contract
env/bin/python prepare.py --root /scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019 --parent /scratch/aparame/Dynamic_VLA/dom100-audited-20261012
