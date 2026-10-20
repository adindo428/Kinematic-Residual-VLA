#!/usr/bin/env bash
#SBATCH --job-name=tracking-table1-summary
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err
set -euo pipefail
cd /scratch/aparame/Dynamic_VLA/runs/tracking_table1_20261008/source
/home/aparame/.conda/envs/dynamicvla/bin/python scripts/summarize_benchmark.py --root "${ROOT:?}"
