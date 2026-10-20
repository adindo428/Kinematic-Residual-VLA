#!/usr/bin/env bash
#SBATCH --job-name=tracking-dom75-velocity
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err
set -euo pipefail
PROJECT=/home/aparame/Research/Dynamic_VLA/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA
cd "$PROJECT"
export HF_HOME=/scratch/aparame/.cache/huggingface
export HF_LEROBOT_HOME=$HF_HOME/lerobot
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=4
export DVLA_N_WORKERS=2 DVLA_DATALOADER_PREFETCH_FACTOR=2 DVLA_PERSISTENT_WORKERS=true
export DVLA_OUTPUT_ROOT=/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_velocity
export WANDB_DIR=/home/aparame/Research/Dynamic_VLA/logs/wandb
export LD_LIBRARY_PATH=/software/spackages/linux-rocky8-x86_64/gcc-9.5.0/cuda-12.1.1-d4ik3tbvtk3ypske36p4ogeulsnsv26p/lib64:${LD_LIBRARY_PATH:-}
unset LEROBOT_HOME
PYTHON=/home/aparame/.conda/envs/dynamicvla/bin/python
experiment=tracking_dom75_full20_velocity_$SLURM_JOB_ID
mkdir -p "$DVLA_OUTPUT_ROOT/provenance" "$WANDB_DIR"
cp configs/dynamicvla_dom_tracking_velocity75.yaml "$DVLA_OUTPUT_ROOT/provenance/$experiment.yaml"
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/train_tracking_velocity.py \
 --cfg configs/dynamicvla_dom_tracking_velocity75.yaml --exp "$experiment" --init-ckpt hzxie/dynamic-vla-DOM
"$PYTHON" -u scripts/verify_tracking_training.py --variant velocity --checkpoint "$DVLA_OUTPUT_ROOT/checkpoints/$experiment"
