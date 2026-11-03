#!/usr/bin/env bash
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=fresh20k-krv-train
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/krv100-fresh20k-h200-20261029/logs/train-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/krv100-fresh20k-h200-20261029/logs/train-%j.err
#SBATCH --gpus-per-node=h200:2
#SBATCH --constraint=gpu_h200&chip_manufacturer_intel
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
cd /scratch/aparame/Dynamic_VLA/krv100-fresh20k-h200-20261029
export PYTHONPATH="$PWD/source"
export WANDB_CONFIG_DIR="$PWD/wandb-config"
mkdir -p "$WANDB_CONFIG_DIR"
secret=/home/aparame/.config/kinematic-residual-vla/secrets.env
if [[ -f "$secret" && ! -L "$secret" && $(stat -c %a "$secret") == 600 ]]; then set -a; source "$secret"; set +a; fi
runner=
forward_stop() { [[ -z "$runner" ]] || kill -USR1 "$runner" || true; }
trap forward_stop USR1 TERM
/home/aparame/.conda/envs/dynamicvla/bin/python launch.py train &
runner=$!
set +e
while true; do wait "$runner"; status=$?; if ! kill -0 "$runner" 2>/dev/null; then break; fi; done
set -e
if [[ $status -eq 75 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$status"
