#!/usr/bin/env bash
#SBATCH --job-name=nvlfix-vlash_baseline
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/restart-repair-20261029-r2/logs/vlash_baseline-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/restart-repair-20261029-r2/logs/vlash_baseline-%j.err
#SBATCH --gpus-per-node=h200:2
#SBATCH --constraint=gpu_h200&graphics
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --exclude=node0102
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
set +e
/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python /scratch/aparame/Dynamic_VLA/restart-repair-20261029-r2/verify.py --name vlash_baseline --gpu
status=$?
set -e
[[ $status -eq 10 ]] && exit 0
[[ $status -eq 0 ]] || exit "$status"
exec bash /scratch/aparame/Dynamic_VLA/h200-microbatch-20261022/vlash_baseline/scripts/vlash_baseline.sbatch
