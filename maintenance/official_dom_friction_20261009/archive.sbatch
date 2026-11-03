#!/usr/bin/env bash
#SBATCH --job-name=dom-friction-hf
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err
set -euo pipefail
export HF_HOME=/scratch/aparame/.cache/huggingface
/home/aparame/.conda/envs/dynamicvla/bin/python /home/aparame/Research/Dynamic_VLA/maintenance/official_dom_friction_20261009/archive.py
