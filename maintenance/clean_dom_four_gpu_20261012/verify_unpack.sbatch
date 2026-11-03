#!/bin/bash -l
#SBATCH --job-name=dom-scp-verify
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/dom-audited-20261012/verify-%j.out
#SBATCH --error=/work/nvme/bivv/aparameshwaran/dynamicvla/dom-audited-20261012/verify-%j.err
set -euo pipefail
python3 /work/nvme/bivv/aparameshwaran/dynamicvla/dom-audited-20261012/verify_unpack.py
