#!/usr/bin/env bash
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=fresh20k-krv-setup
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/krv100-fresh20k-h200-20261029/logs/setup-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/krv100-fresh20k-h200-20261029/logs/setup-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
cd /scratch/aparame/Dynamic_VLA/krv100-fresh20k-h200-20261029
export PYTHONPATH="$PWD/source"
export WANDB_CONFIG_DIR="$PWD/wandb-config"
mkdir -p "$WANDB_CONFIG_DIR"
secret=/home/aparame/.config/kinematic-residual-vla/secrets.env
if [[ -f "$secret" && ! -L "$secret" && $(stat -c %a "$secret") == 600 ]]; then set -a; source "$secret"; set +a; fi
/home/aparame/.conda/envs/dynamicvla/bin/python cpu_preflight.py
