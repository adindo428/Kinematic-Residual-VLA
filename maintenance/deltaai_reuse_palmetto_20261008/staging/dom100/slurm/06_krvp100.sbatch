#!/usr/bin/env bash
#SBATCH --job-name=krvp100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=krvp100-%j.out
#SBATCH --error=krvp100-%j.err

set -euo pipefail
PROJECT_DIR=${DVLA_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "${PROJECT_DIR}"
exec bash scripts/run_dom100.sh train krvp100
