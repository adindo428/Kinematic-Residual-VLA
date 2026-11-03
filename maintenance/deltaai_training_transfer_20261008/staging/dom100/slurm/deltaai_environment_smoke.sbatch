#!/usr/bin/env bash
#SBATCH --job-name=dom-vla-env
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
env_root=/work/nvme/${DVLA_ACCESS_PROJECT}/${USER}/dynamicvla
venv_dir=${env_root}/venv-dynamicvla-torch212
mkdir -p "${env_root}"
if [[ ! -f "${venv_dir}/pyvenv.cfg" ]]; then
  python -m venv --system-site-packages "${venv_dir}"
fi
source "${venv_dir}/bin/activate"
python -m pip install --disable-pip-version-check \
  'av>=14.2.0' 'cmake>=3.29.0.1' 'deepdiff>=7,<9' \
  'draccus==0.10.0' 'easydict' 'flask>=3,<4' \
  'gymnasium>=0.29.1,<1' 'imageio>=2.34,<3' \
  'imageio-ffmpeg' 'jsonlines>=4' 'pynput>=1.7.7' \
  'pyserial>=3.5' 'rerun-sdk>=0.22,<0.23' 'termcolor>=2.4,<4' \
  'num2words' \
  --constraint "${SLURM_SUBMIT_DIR}/configs/deltaai_torch_constraints.txt"
python -m pip install --disable-pip-version-check --no-deps \
  'lerobot==0.3.3' 'torchcodec==0.16.0'

python - <<'PY'
import sys
import torch
import torchcodec
import torchcodec.decoders
import lerobot
import easydict
import draccus
import torchvision
import num2words
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("TorchCodec:", torchcodec.__version__)
if not torch.__version__.startswith("2.12.0+cu130"):
    raise SystemExit("Wrong PyTorch version in the project environment")
print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No GPU visible in the smoke allocation")
print("LeRobot, TorchCodec, EasyDict, Draccus imports: OK")
PY
