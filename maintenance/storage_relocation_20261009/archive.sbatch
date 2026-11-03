#!/usr/bin/env bash
#SBATCH --job-name=results-hf-archive
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/aparame/storage_relocation_20261009/logs/%x-%j.out
#SBATCH --error=/scratch/aparame/storage_relocation_20261009/logs/%x-%j.err
set -euo pipefail
export HF_HOME=/scratch/aparame/.cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
export OMP_NUM_THREADS=2
CODE=/home/aparame/Research/Dynamic_VLA/maintenance/storage_relocation_20261009
args=()
if [[ "${WAIT_FOR_SUMMARY:-0}" == 1 ]]; then args+=(--wait-for-summary); fi
/home/aparame/.conda/envs/dynamicvla/bin/python -u "$CODE/archive_results.py" "${args[@]}"
