#!/usr/bin/env bash
#SBATCH --job-name=dom-vla-pair
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200g
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this file through sbatch}"
: "${DVLA_ACCESS_PROJECT:?Export the actual DeltaAI project serial code}"
: "${DVLA_HF_CHECKPOINT_REPO:?Export the Hugging Face model repository ID}"
: "${DVLA_SECRETS_FILE:?Set an absolute path to a mode-600 project secrets file}"
: "${DVLA_WANDB_ENTITY:?Set the W&B entity for this project}"
: "${DVLA_PYTHON_ENV:?Set the validated project environment path}"

cd "${SLURM_SUBMIT_DIR:?Submit from the DynamicVLA project root}"
export DVLA_ENV_SCRIPT="${PWD}/scripts/deltaai_env.sh"
source "${DVLA_ENV_SCRIPT}"
python scripts/check_train_readiness.py
export NPROC_PER_NODE=4
export DVLA_N_WORKERS=${DVLA_N_WORKERS:-2}
export DVLA_BATCH_SIZE=${DVLA_BATCH_SIZE:-16}
export DVLA_GRAD_ACCUM_STEPS=${DVLA_GRAD_ACCUM_STEPS:-16}
export RUN_PREFETCH=${RUN_PREFETCH:-0}
export RUN_OFFLINE_COMPARE=0
export AUTO_RESUME=1
export BASELINE_EXP=dom-vla-Delta-AI
export RESIDUAL_EXP=residual-kinematic-vla-Delta-AI
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

bash scripts/train_full_finetune_speed_ood_pair.sh
