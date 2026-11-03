#!/usr/bin/env bash
#SBATCH --job-name=storage-cache-move
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/aparame/storage_relocation_20261009/logs/%x-%j.out
#SBATCH --error=/scratch/aparame/storage_relocation_20261009/logs/%x-%j.err
set -euo pipefail
CODE=/home/aparame/Research/Dynamic_VLA/maintenance/storage_relocation_20261009
PYTHON=/home/aparame/.conda/envs/dynamicvla/bin/python
"$PYTHON" "$CODE/relocate.py" --cache pip
"$PYTHON" "$CODE/relocate.py" --cache uv
