#!/usr/bin/env bash
#SBATCH --job-name=krv100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --constraint="gpu_h100|gpu_h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=krv100-ddp-%j.out
#SBATCH --error=krv100-ddp-%j.err
set -euo pipefail
cd "${DVLA_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash scripts/run_dom100_distributed.sh krv100
