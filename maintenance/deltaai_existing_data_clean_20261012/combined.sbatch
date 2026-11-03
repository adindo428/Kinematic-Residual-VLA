#!/usr/bin/env bash
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --job-name=domv-verify-train
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-audited-20261012/logs/train-%j.out
#SBATCH --error=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-audited-20261012/logs/train-%j.err
set -euo pipefail
source /work/nvme/bivv/aparameshwaran/dynamicvla/setup-20261008/activate-training.sh
export DVLA_PROJECT_DIR=/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100
export DOM100_ROOT=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-audited-20261012
export DOM100_DDP_GPUS=4 GLOO_SOCKET_IFNAME=lo DOM100_VERIFY_THREADS=16
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
HERE=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-existing-data-domv-20261012
cd "$DVLA_PROJECT_DIR"
# One queue wait: verify, check readiness, then train without releasing the GPUs.
# On checkpoint requeue, completed verification and runtime approval are reused.
export WANDB_MODE=disabled
timeout --signal=TERM --kill-after=60s 80m python "$HERE/prepare_combined.py"
export DOM100_COMMAND=train DOM100_RUNTIME_FILE="$DOM100_ROOT/selected-runtime.json"
export WANDB_MODE=online
exec bash scripts/run_dom100_distributed.sh domv100
