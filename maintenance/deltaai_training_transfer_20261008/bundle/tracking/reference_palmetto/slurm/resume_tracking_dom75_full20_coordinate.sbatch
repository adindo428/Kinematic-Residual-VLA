#!/usr/bin/env bash
#SBATCH --job-name=tracking-dom75-coordinate-resume
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
source /home/aparame/Research/Dynamic_VLA/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA/scripts/palmetto_env.sh
cd /home/aparame/Research/Dynamic_VLA/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA
export DVLA_OUTPUT_ROOT=/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=4
export DVLA_N_WORKERS=2
export DVLA_DATALOADER_PREFETCH_FACTOR=2
export DVLA_PERSISTENT_WORKERS=true

torchrun --standalone --nnodes=1 --nproc_per_node=1 run.py   --cfg /scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/provenance/tracking_dom75_full20_coordinate.yaml   --exp tracking_dom75_full20_coordinate_resume_e14   --ckpt /scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/checkpoints/tracking_dom75_full20_coordinate_16623338
