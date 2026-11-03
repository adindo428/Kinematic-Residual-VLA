#!/usr/bin/env bash
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --job-name=dom100-20k-2n
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=/work/nvme/bicn/anayak2/dynamicvla/dom100-20k-gh200-20261028/logs/train-%j.out
#SBATCH --error=/work/nvme/bicn/anayak2/dynamicvla/dom100-20k-gh200-20261028/logs/train-%j.err
set -euo pipefail
source /work/nvme/bicn/anayak2/dynamicvla/dom100-20k-gh200-20261028/activate.sh
cd "$DVLA_PROJECT_DIR"
exec python "$DOM_BASELINE_ROOT/run_job.py"
