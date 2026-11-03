#!/usr/bin/env bash
#SBATCH --job-name=smol-closedloop-pilot
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-20261025/logs/pilot-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-20261025/logs/pilot-%j.err
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-20261025/scripts
cd /scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-20261025
/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python /scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-20261025/run.py --stage /scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-20261025 --root /scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-20261025/results --mode pilot
