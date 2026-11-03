#!/usr/bin/env bash
#SBATCH --job-name=dom-pilots
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=dom-pilots-%j.out

set -euo pipefail
cd "${DVLA_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash scripts/run_dom100.sh pilots
