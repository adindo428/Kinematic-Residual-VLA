#!/bin/bash -l
#SBATCH --job-name=pi05_cpu_pipeline_diag
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --constraint=gpu_h200
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/pi05-cpu-pipeline-20261021/logs/benchmark-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/pi05-cpu-pipeline-20261021/logs/benchmark-%j.err
set -euo pipefail
cd /scratch/aparame/Dynamic_VLA/pi05-cpu-pipeline-20261021
export HF_HOME=/scratch/aparame/.cache/huggingface
export PYTHONPATH=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020:/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/upstream-vlash
export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 benchmark.py
