#!/usr/bin/env bash
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --job-name=domv-profiled-gh200
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-control-ddp-20261011/logs/train-%j.out
#SBATCH --error=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-control-ddp-20261011/logs/train-%j.err
set -euo pipefail
source /work/nvme/bivv/aparameshwaran/dynamicvla/setup-20261008/activate-training.sh
export DVLA_PROJECT_DIR=/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261011-control-gloo/dom100
export DOM100_ROOT=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-control-ddp-20261011
export DOM100_DDP_GPUS=4 GLOO_SOCKET_IFNAME=lo
cd "$DVLA_PROJECT_DIR"
python - <<'PY_CHECK'
import os,json
from pathlib import Path
import torch
import lerobot
from core.dom_training import load_family
from utils.dom_experiment import file_digest
m=load_family(Path(os.environ['DOM100_ROOT']))
assert torch.cuda.device_count()==4
assert all('GH200' in torch.cuda.get_device_name(i) for i in range(4))
assert file_digest(Path(m['initialization'])/'model.safetensors')==m['weights_sha256']
s=json.loads((Path(os.environ['DVLA_WORK_ROOT'])/'setup-20261008/hzxie--DOM-status.json').read_text())
assert s['state']=='complete' and s['revision']==m['dataset_revision']
PY_CHECK
export DOM100_COMMAND=train
export DOM100_RUNTIME_FILE="$DOM100_ROOT/selected-runtime.json"
export WANDB_MODE=online
exec bash scripts/run_dom100_distributed.sh domv100
