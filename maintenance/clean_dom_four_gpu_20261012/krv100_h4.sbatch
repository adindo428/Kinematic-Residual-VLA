#!/bin/bash -l
#SBATCH --job-name=krv100-h4
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:4
#SBATCH --constraint=gpu_h100
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=/scratch/aparame/Dynamic_VLA/dom100-audited-20261012/logs/krv100-h4-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/dom100-audited-20261012/logs/krv100-h4-%j.err
set -euo pipefail
export DVLA_PROJECT_DIR=/scratch/aparame/Dynamic_VLA/third_party/dom100-ddp-video-clean-20261012
export DOM100_ROOT=/scratch/aparame/Dynamic_VLA/dom100-audited-20261012
export DOM100_DDP_GPUS=4 DOM100_DDP_MICROBATCH=44 DOM100_DDP_WORKERS=1
# 44 x 4 GPUs x 2 accumulation = 352; trainer derives accumulation.
cd "$DVLA_PROJECT_DIR"
exec bash scripts/run_dom100_allocation_candidate.sh krv100
