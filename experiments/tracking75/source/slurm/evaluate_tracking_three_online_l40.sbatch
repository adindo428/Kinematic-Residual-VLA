#!/usr/bin/env bash
#SBATCH --job-name=krvla-three-online
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=l40:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err
set -euo pipefail
export PROJECT=${PROJECT:-/home/aparame/Research/Dynamic_VLA/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA}
PYTHON=${PYTHON:-/home/aparame/.conda/envs/dynamicvla/bin/python}
comparison_root=${OUTPUT_ROOT:-/home/aparame/Research/Dynamic_VLA/RESULTS/tracking_current_eval}
cd "$PROJECT"
launcher=slurm/evaluate_tracking_case1_online_l40.sbatch
mode=${MODE:-full}
if [[ "$mode" != full ]]; then
 MODE="$mode" OUTPUT_ROOT="$comparison_root/$mode" bash "$launcher"
 exit 0
fi
MODE=policy-smoke OUTPUT_ROOT="$comparison_root/policy_smoke" bash "$launcher"
MODE=smoke OUTPUT_ROOT="$comparison_root/online_smoke" bash "$launcher"
"$PYTHON" - "$comparison_root/online_smoke/$SLURM_JOB_ID/summary.json" <<'PY'
import json,sys
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text())
assert s['complete'] and s['pairing_valid'], 'Smoke encountered an infrastructure/pairing failure'
assert s['recorded_rollouts']==9 and len(s['comparison_videos'])==3
assert all(Path(v['path']).is_file() and v['frames']==50 for v in s['comparison_videos'])
# A legitimate policy termination is an evaluation outcome, not a reason to censor a model.
print('Three-model live smoke passed; policy terminations:',s['policy_terminations'])
PY
MODE=full OUTPUT_ROOT="$comparison_root/online" bash "$launcher"
"$PYTHON" - "$comparison_root/online/$SLURM_JOB_ID/summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert s['complete'] and s['recorded_rollouts']==90 and s['pairing_valid']
print('All 90 trials evaluated; policy terminations:',s['policy_terminations'])
PY
