#!/bin/bash -l
#SBATCH --job-name=pi05_h200_handoff_verify
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-control-20261020/logs/verify-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-control-20261020/logs/verify-%j.err
set -euo pipefail
cd /scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-opt-r2-20261020
export HF_HOME=/scratch/aparame/.cache/huggingface
export PYTHONPATH=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/upstream-vlash
export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

env/bin/python /scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-control-20261020/handoff.py verify --config-sha d3f4f1c7f7eaea5249585a20a7b9780d442d693f1e464cb302e4160435c6a055 --controller-sha db557049687bc22dd638e1f0a02600e36c074494574aa39865cbcf30e33b6ac4
