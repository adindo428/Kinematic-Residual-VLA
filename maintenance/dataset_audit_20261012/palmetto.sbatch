#!/bin/bash -l
#SBATCH --job-name=dom-audit
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-15%4
#SBATCH --output=/scratch/aparame/Dynamic_VLA/dataset-audit-20261012/audit-%A_%a.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/dataset-audit-20261012/audit-%A_%a.err
set -euo pipefail
module load gcc/12.3.0 cuda/12.3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dynamicvla
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python /scratch/aparame/Dynamic_VLA/dataset-audit-20261012/audit.py --family /scratch/aparame/Dynamic_VLA/dom100-ddp-video-clean-20261012 --output /scratch/aparame/Dynamic_VLA/dataset-audit-20261012/${AUDIT_OUTPUT:-results} --shard "${SLURM_ARRAY_TASK_ID}" --workers 8 ${AUDIT_LIMIT:+--limit "$AUDIT_LIMIT"}
