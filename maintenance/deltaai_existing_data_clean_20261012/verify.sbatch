#!/bin/bash -l
#SBATCH --job-name=dom-existing-verify
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-existing-data-domv-20261012/verify-%j.out
#SBATCH --error=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-existing-data-domv-20261012/verify-%j.err
set -euo pipefail
HERE=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-existing-data-domv-20261012
FAMILY=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-audited-20261012
exec 9>"$HERE/freeze.lock"
flock 9
if [[ ! -e "$FAMILY" ]]; then
  python3 "$HERE/freeze_family.py" /work/nvme/bivv/aparameshwaran/dynamicvla/domv-video-clean-ddp-20261012 "$FAMILY" /u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100 "$HERE/audit-receipts"
fi
flock -u 9
python3 "$HERE/verify_existing.py"
