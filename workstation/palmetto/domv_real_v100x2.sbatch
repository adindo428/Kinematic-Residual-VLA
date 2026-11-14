#!/usr/bin/env bash
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=v100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=domv-real-v100x2
#SBATCH --requeue
#SBATCH --output=/scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114/logs/train-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114/logs/train-%j.err
set -euo pipefail
DOMV_ROOT=/scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114
source "$DOMV_ROOT/env.sh"
# Do not inherit the one-core affinity of the CPU-control submission step.
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
cd "$DOMV_ROOT/app"
# Disposable distributed check: strict original-base load, native sampler parity,
# FP16/GradScaler, finite updates, and synchronized parameter checksums.
srun --ntasks=1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=none \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  workstation/train_dom_posttraining_ddp.py --contract "$DOMV_ROOT/launch_contract.json" \
  --output "$DOMV_ROOT/disposable-check" --preflight
resume_flags=()
if [[ -f "$DOMV_ROOT/domv/training_state.pt" ]]; then
  resume_flags=(--resume)
elif [[ -d "$DOMV_ROOT/domv" ]]; then
  mv "$DOMV_ROOT/domv" "$DOMV_ROOT/incomplete-domv-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}"
fi
# This is a new process. Disposable preflight updates are never training inputs.
srun --ntasks=1 --cpus-per-task=16 --cpu-bind=none --gpu-bind=none \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  workstation/train_dom_posttraining_ddp.py --contract "$DOMV_ROOT/launch_contract.json" \
  --output "$DOMV_ROOT" "${resume_flags[@]}"
