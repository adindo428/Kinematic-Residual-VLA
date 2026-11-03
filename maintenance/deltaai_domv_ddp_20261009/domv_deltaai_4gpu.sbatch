#!/usr/bin/env bash
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=domv-ddp-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-ddp-es-20261009/logs/train-%j.out
#SBATCH --error=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-ddp-es-20261009/logs/train-%j.err
set -euo pipefail
source /work/nvme/bivv/aparameshwaran/dynamicvla/setup-20261008/activate-training.sh
export DVLA_PROJECT_DIR=/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261009T165212Z/dom100
export DOM100_ROOT=/work/nvme/bivv/aparameshwaran/dynamicvla/domv-ddp-es-20261009
export DOM100_DDP_GPUS=4 DOM100_DDP_MICROBATCH=44 DOM100_DDP_WORKERS=1
export WANDB_MODE=online
cd "$DVLA_PROJECT_DIR"
python - <<'PY'
import json,os
from pathlib import Path
import torch
from core.dom_training import load_family
from utils.dom_experiment import digest,file_digest
root=Path(os.environ['DOM100_ROOT'])
manifest=load_family(root)
recipe=json.loads((root/'recipe.json').read_text())
check=dict(recipe);fingerprint=check.pop('fingerprint')
assert digest(check)==fingerprint and recipe['manifest']==manifest['fingerprint']
assert manifest['prediction_horizon']==manifest['execution_horizon']==20
assert torch.cuda.device_count()==4
assert all('GH200' in torch.cuda.get_device_name(i) for i in range(4))
assert file_digest(Path(manifest['initialization'])/'model.safetensors')==manifest['weights_sha256']
status=json.loads((Path(os.environ['DVLA_WORK_ROOT'])/'setup-20261008/hzxie--DOM-status.json').read_text())
assert status['state']=='complete' and status['revision']==manifest['dataset_revision']
print(json.dumps({'job':os.environ['SLURM_JOB_ID'],'variant':'domv100','gpus':4,'backend':'DDP/NCCL',
 'family':str(root),'manifest':manifest['fingerprint'],'recipe':recipe,'microbatch_per_gpu':44,
 'accumulation_microsteps_per_update':2,'purpose':'paired_DOM_plus_velocity_training'}),flush=True)
PY
exec bash scripts/run_dom100_distributed.sh domv100
