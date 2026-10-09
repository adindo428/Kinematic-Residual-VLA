#!/usr/bin/env bash
#SBATCH --job-name=tracking-coord-vel75
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err
set -euo pipefail
PROJECT=${PROJECT:-/home/aparame/Research/Dynamic_VLA/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA}
cd "$PROJECT"
# Isolate the run from inherited experimental overrides.
for name in ${!DVLA_@}; do unset "$name"; done
export HF_HOME=/scratch/aparame/.cache/huggingface HF_HUB_OFFLINE=1
export HF_LEROBOT_HOME=$HF_HOME/lerobot
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=4 PYTHONHASHSEED=42
export DVLA_N_WORKERS=2 DVLA_DATALOADER_PREFETCH_FACTOR=2 DVLA_PERSISTENT_WORKERS=true
export DVLA_OUTPUT_ROOT=/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_coordinate_velocity
export WANDB_DIR=/home/aparame/Research/Dynamic_VLA/logs/wandb
export LD_LIBRARY_PATH=/software/spackages/linux-rocky8-x86_64/gcc-9.5.0/cuda-12.1.1-d4ik3tbvtk3ypske36p4ogeulsnsv26p/lib64:${LD_LIBRARY_PATH:-}
unset LEROBOT_HOME
PYTHON=/home/aparame/.conda/envs/dynamicvla/bin/python
experiment=tracking_dom75_full20_coordinate_velocity_$SLURM_JOB_ID
checkpoint=$DVLA_OUTPUT_ROOT/checkpoints/$experiment
[[ ! -e "$checkpoint" ]] || { echo "Refusing to overwrite $checkpoint" >&2; exit 2; }
mkdir -p "$DVLA_OUTPUT_ROOT/provenance" "$WANDB_DIR"
cp configs/tracking_dom75_full20_coordinate_velocity.yaml "$DVLA_OUTPUT_ROOT/provenance/$experiment.yaml"
smoke=/home/aparame/Research/Dynamic_VLA/RESULTS/tracking_coordinate_velocity_verification/$SLURM_JOB_ID
"$PYTHON" -u scripts/smoke_tracking_coordinate_velocity.py \
 --config configs/tracking_dom75_full20_coordinate_velocity.yaml --output "$smoke" \
 --previous-config /scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/resume_outputs/checkpoints/tracking_dom75_full20_coordinate_16623338/config.json
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/train_tracking_velocity.py \
 --cfg configs/tracking_dom75_full20_coordinate_velocity.yaml --exp "$experiment" --init-ckpt hzxie/dynamic-vla-DOM
"$PYTHON" -u scripts/verify_tracking_training.py --variant coordinate --require-neural-velocity --checkpoint "$checkpoint"
