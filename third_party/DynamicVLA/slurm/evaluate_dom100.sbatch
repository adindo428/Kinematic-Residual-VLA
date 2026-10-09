#!/usr/bin/env bash
#SBATCH --job-name=dom-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=dom-eval-%j.out

set -euo pipefail
cd "${DVLA_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/dom100_env.sh
python scripts/dom_closed_loop.py evaluate --family-root "$1" --request "$2"
