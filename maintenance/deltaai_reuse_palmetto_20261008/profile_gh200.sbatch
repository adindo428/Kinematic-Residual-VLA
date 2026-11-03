#!/bin/bash
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=domprof-gh200
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/dom100-palmetto-h20/profiling/gh200-%j.out
set -euo pipefail
source /work/nvme/bivv/aparameshwaran/dynamicvla/setup-20261008/activate-training.sh
export WANDB_MODE=disabled
cd "$DVLA_PROJECT_DIR"
OUTPUT="$DOM100_ROOT/profiling/gh200-$SLURM_JOB_ID"
mkdir -p "$OUTPUT"
python - <<'PY'
import json, os
from pathlib import Path
import torch
from core.dom_training import load_family
root=Path(os.environ['DOM100_ROOT'])
manifest=load_family(root)
status=json.loads((Path(os.environ['DVLA_WORK_ROOT'])/'setup-20261008/hzxie--DOM-status.json').read_text())
assert status['state']=='complete' and status['revision']==manifest['dataset_revision']
assert manifest['prediction_horizon']==manifest['execution_horizon']==20
assert torch.cuda.device_count()==1
assert 'GH200' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print(json.dumps({'job':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name(0),'manifest':manifest['fingerprint'],'purpose':'profiling_only','effective_batch':352,'prediction_horizon':20,'execution_horizon':20}),flush=True)
PY
python -u scripts/dom100.py profile --family-root "$DOM100_ROOT" --output "$OUTPUT" > "$OUTPUT/controller.log" 2>&1 &
PROFILE_PID=$!
trap 'kill "$PROFILE_PID" 2>/dev/null || true' TERM INT
wait "$PROFILE_PID"
python - <<'PY'
import json, os
from pathlib import Path
from core.dom_profiling import gpu_key
root=Path(os.environ['DOM100_ROOT'])
path=root/'profiles'/f'{gpu_key()}.json'
profile=json.loads(path.read_text())
print(json.dumps({'profiling_complete':True,'profile':str(path),'runtime':profile['runtime'],'all_variants_verified':profile['all_variants_verified'],'training_launched':False}),flush=True)
PY
