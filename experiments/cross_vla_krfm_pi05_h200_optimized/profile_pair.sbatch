#!/bin/bash -l
#SBATCH --job-name=pi05_h200_opt_pair
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --constraint=gpu_h200
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-opt-20261020/logs/profile-pair-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-opt-20261020/logs/profile-pair-%j.err
set -euo pipefail
cd /scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-opt-20261020
export HF_HOME=/scratch/aparame/.cache/huggingface
export PYTHONPATH=/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/upstream-vlash
export LD_LIBRARY_PATH=/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
env/bin/python - <<'CHECK'
import json
from pathlib import Path
from common import sha
p=Path('.')
m=json.loads((p/'source-manifest.json').read_text())
for n,h in m['files'].items():
    assert sha(p/n)==h,n
CHECK
env/bin/python autotune.py --root /scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-opt-20261020 --recipe pi05_baseline
env/bin/python autotune.py --root /scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-opt-20261020 --recipe pi05_kr
env/bin/python - <<'CHECK'
import json,os
from pathlib import Path
from common import atomic
p=Path('.')
a={n:json.loads((p/'profiles'/f'{n}.json').read_text()) for n in ['pi05_baseline','pi05_kr']}
assert all(v['approved'] and v['decoded_optimizer_updates']==200 for v in a.values())
atomic(p/'profile-complete.json',{'complete':True,'job_id':os.environ['SLURM_JOB_ID'],'profiles':a,'production_submitted':False})
CHECK
