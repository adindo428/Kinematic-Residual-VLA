#!/usr/bin/env bash
#SBATCH --job-name=official-dom-summary
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err
set -euo pipefail
STAGE=/scratch/aparame/Dynamic_VLA/runs/official_dom_table1_20261009
ROOT=/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_table1/20261009
export MPLCONFIGDIR=$STAGE/runtime/matplotlib
mkdir -p "$MPLCONFIGDIR"
/home/aparame/.conda/envs/dynamicvla/bin/python "$STAGE/scripts/report.py" --root "$ROOT"
