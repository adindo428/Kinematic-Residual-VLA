#!/usr/bin/env bash
#SBATCH --job-name=krv100-h2x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --constraint=gpu_h200
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=krv100-h2x8-%j.out
#SBATCH --error=krv100-h2x8-%j.err
set -euo pipefail
cd "${DVLA_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export DOM100_DDP_GPUS=8 DOM100_DDP_MICROBATCH=44 DOM100_DDP_WORKERS=1
exec bash scripts/run_dom100_allocation_candidate.sh krv100
