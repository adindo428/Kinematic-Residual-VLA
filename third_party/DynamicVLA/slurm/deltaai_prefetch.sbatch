#!/usr/bin/env bash
#SBATCH --job-name=dom-vla-data
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the DynamicVLA root}"
export DVLA_ACCESS_PROJECT=${DVLA_ACCESS_PROJECT:-bivv}
export DVLA_PYTHON_ENV=${DVLA_PYTHON_ENV:-/work/nvme/${DVLA_ACCESS_PROJECT}/${USER}/dynamicvla/venv-dynamicvla-torch212}
export DVLA_SECRETS_FILE=${DVLA_SECRETS_FILE:-/u/${USER}/.config/kinematic-residual-vla/secrets.env}
source scripts/deltaai_env.sh
export DYNAMICVLA_PREFETCH_MODE=file
export DYNAMICVLA_PREFETCH_STATUS_INTERVAL=300
export DYNAMICVLA_PREFETCH_RETRIES=${DYNAMICVLA_PREFETCH_RETRIES:-8}
export DYNAMICVLA_PREFETCH_FILE_DELAY=${DYNAMICVLA_PREFETCH_FILE_DELAY:-2}
export DYNAMICVLA_PREFETCH_RATE_LIMIT_DELAY=${DYNAMICVLA_PREFETCH_RATE_LIMIT_DELAY:-120}
python scripts/prefetch_eval_splits.py \
  --cfg configs/dynamicvla_dom_full_finetune_speed_ood_0p10_0p18.yaml \
  --splits train ood_val
python scripts/check_train_readiness.py
