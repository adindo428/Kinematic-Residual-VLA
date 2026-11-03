#!/bin/bash -l
#SBATCH --job-name=a100r2_vlash_kr-prof
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:2
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=32
#SBATCH --mem=224G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-r2-20261019/logs/vlash_kr-profile-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-r2-20261019/logs/vlash_kr-profile-%j.err
set -euo pipefail
cd /scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-r2-20261019
export HF_HOME=/scratch/aparame/.cache/huggingface
export PYTHONPATH=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/upstream-vlash
export LD_LIBRARY_PATH=/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_START_METHOD=thread
export WANDB_INIT_TIMEOUT=45
export WANDB_DIR=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-r2-20261019/runs/vlash_kr
mkdir -p "$WANDB_DIR"
trap 'pkill -USR1 -P "$runner" || true; wait "$runner" || true' USR1 TERM
rm -f /scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-r2-20261019/runs/vlash_kr/stopped.json
set +e
env/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 train.py profile --root /scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-r2-20261019 --recipe vlash_kr &
runner=$!
wait "$runner"
result=$?
set -e
exit "$result"
