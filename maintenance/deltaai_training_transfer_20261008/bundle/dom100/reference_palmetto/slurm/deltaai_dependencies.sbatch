#!/usr/bin/env bash
#SBATCH --job-name=dom-vla-deps
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from DynamicVLA root}"
export DVLA_ACCESS_PROJECT=${DVLA_ACCESS_PROJECT:-bivv}
module load python/miniforge3_pytorch/2.12.0
source /work/nvme/${DVLA_ACCESS_PROJECT}/${USER}/dynamicvla/venv-dynamicvla-torch212/bin/activate
python -m pip install --disable-pip-version-check 'num2words'
python - <<'PY'
import torch
import torchcodec.decoders
import num2words
import utils.datasets
import core.train
print("DynamicVLA training imports OK; Torch:", torch.__version__)
print("CUDA visible:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
PY
