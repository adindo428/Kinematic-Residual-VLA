#!/bin/bash -l
#SBATCH --job-name=ep5_smolvla_kr_train
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:2
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=32
#SBATCH --mem=224G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/logs/smolvla_kr-train-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/logs/smolvla_kr-train-%j.err

#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
set -euo pipefail
cd /scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020
export HF_HOME=/scratch/aparame/.cache/huggingface
export PYTHONPATH=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/upstream-vlash
export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export WANDB_START_METHOD=thread
export WANDB_INIT_TIMEOUT=45
export WANDB_DIR=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/runs/smolvla_kr
mkdir -p "$WANDB_DIR"
env/bin/python /scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/control/handoff.py --recipe smolvla_kr --manifest-sha b3a1272058201cfe48832a7e99d248ba045106e61fc35e6b1aab5bf90639396d
trap 'pkill -USR1 -P "$runner" || true; wait "$runner" || true' USR1 TERM
rm -f /scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/runs/smolvla_kr/stopped.json
set +e
env/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 train.py train --root /scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020 --recipe smolvla_kr &
runner=$!
wait "$runner"
status=$?
set -e
if [[ -f /scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/runs/smolvla_kr/stopped.json && ! -f /scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/runs/smolvla_kr/completed.json ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
