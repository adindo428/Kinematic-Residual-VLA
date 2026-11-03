#!/usr/bin/env bash
#SBATCH --job-name=dom-grasp-diag
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=l40s:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err
set -euo pipefail
export HF_HOME=/scratch/aparame/.cache/huggingface HF_HUB_OFFLINE=1
export HF_LEROBOT_HOME=$HF_HOME/lerobot
export LD_LIBRARY_PATH=/software/spackages/linux-rocky8-x86_64/gcc-9.5.0/cuda-12.1.1-d4ik3tbvtk3ypske36p4ogeulsnsv26p/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false TERM=xterm
unset LEROBOT_HOME
STAGE=/scratch/aparame/Dynamic_VLA/runs/official_dom_diagnostic_20261009
ROOT=/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_diagnostic/20261009
if [[ "${DIAG_DUAL:-0}" == 1 ]]; then
    /home/aparame/.conda/envs/dynamicvla/bin/python "$STAGE/scripts/run.py" --stage "$STAGE" --root "$ROOT" --gpus 2 --labels gripper_aligned_dual
else
    /home/aparame/.conda/envs/dynamicvla/bin/python "$STAGE/scripts/run.py" --stage "$STAGE" --root "$ROOT"
fi
