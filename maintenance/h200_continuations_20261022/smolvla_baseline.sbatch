#!/usr/bin/env bash
#SBATCH --job-name=h2-cont-smolvla_baseline
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/h200-continuations-20261022/logs/smolvla_baseline-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/h200-continuations-20261022/logs/smolvla_baseline-%j.err
#SBATCH --gpus-per-node=h200:2
#SBATCH --constraint=gpu_h200
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>/scratch/aparame/Dynamic_VLA/h200-continuations-20261022/smolvla_baseline.lock
flock -n 9 || exit 1
set +e
/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python /scratch/aparame/Dynamic_VLA/h200-continuations-20261022/guard.py --spec /scratch/aparame/Dynamic_VLA/h200-continuations-20261022/specs/smolvla_baseline.json
status=$?
set -e
if [[ $status -eq 10 ]]; then exit 0; fi
[[ $status -eq 0 ]] || exit "$status"
exec bash /scratch/aparame/Dynamic_VLA/main8-h200-only-20261022/scripts/smolvla_baseline.sbatch
