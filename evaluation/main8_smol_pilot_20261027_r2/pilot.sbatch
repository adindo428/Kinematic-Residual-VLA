#!/usr/bin/env bash
#SBATCH --job-name=smol-closedloop-r2
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-r2-20261027/logs/pilot-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-r2-20261027/logs/pilot-%j.err
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-r2-20261027/scripts
cd /scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-r2-20261027
/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python /scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-r2-20261027/run.py --stage /scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-r2-20261027 --root /scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-r2-20261027/results --mode pilot
