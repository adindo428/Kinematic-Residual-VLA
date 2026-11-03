#!/usr/bin/env bash
#SBATCH --job-name=coord-vel-closedloop
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=l40s:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err
set -euo pipefail
WORKSPACE=/home/aparame/Research/Dynamic_VLA
PROJECT=/scratch/aparame/Dynamic_VLA/runs/tracking_coordinate_velocity_eval/source_20261007
COLLECTOR=/scratch/aparame/Dynamic_VLA/runs/tracking_coordinate_velocity_eval/collector_20261007
PYTHON=/home/aparame/.conda/envs/dynamicvla/bin/python
PREVIOUS=$WORKSPACE/RESULTS/tracking_comparison_four_75epoch/Coordinate_NOT_Conditioned_Velocity/live_tracking
CHECKPOINT=/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_coordinate_velocity/checkpoints/tracking_dom75_full20_coordinate_velocity_16748131
ROOT=$WORKSPACE/RESULTS/tracking_comparison_four_75epoch/Coordinate_Conditioned_Velocity/$SLURM_JOB_ID
OUTPUT=$ROOT/live_tracking
COMBINED=$ROOT/comparison_with_existing
SIF_FILE=/scratch/aparame/isaaclab.sif
ISAAC_DEPS=/scratch/aparame/Dynamic_VLA/isaac_eval_deps/python
export HF_HOME=/scratch/aparame/.cache/huggingface HF_HUB_OFFLINE=1
export HF_LEROBOT_HOME=$HF_HOME/lerobot
export LD_LIBRARY_PATH=/software/spackages/linux-rocky8-x86_64/gcc-9.5.0/cuda-12.1.1-d4ik3tbvtk3ypske36p4ogeulsnsv26p/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
unset LEROBOT_HOME
cd "$PROJECT"
[[ ! -e "$ROOT" ]] || { echo "Refusing to overwrite $ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT" "$ROOT/provenance"
cp "$CHECKPOINT/training_ready.json" "$ROOT/provenance/coordinate_velocity_training_ready.json"
cp source_manifest.json "$ROOT/provenance/source_manifest.json"
"$PYTHON" -u scripts/prepare_coordinate_velocity_eval.py --previous "$PREVIOUS" --checkpoint "$CHECKPOINT" \
 --collector "$COLLECTOR" --output "$OUTPUT/manifest.json"
"$PYTHON" -u scripts/tracking_online_policy.py --smoke --coordinate-velocity "$CHECKPOINT" --output "$ROOT/provenance"
command -v apptainer >/dev/null
[[ -f "$SIF_FILE" && -d "$PREVIOUS/sim_deps" ]]
mkdir -p "$OUTPUT/kit_data" "$OUTPUT/kit_cache" "$OUTPUT/sim_tmp"
ln -s "$PREVIOUS/sim_deps" "$OUTPUT/sim_deps"
container=(apptainer exec --cleanenv --nv -B "$WORKSPACE:$WORKSPACE" -B /scratch/aparame:/scratch/aparame
 -B "$OUTPUT/kit_data:/isaac-sim/kit/data" -B "$OUTPUT/kit_cache:/isaac-sim/kit/cache" -B "$OUTPUT/sim_tmp:/tmp"
 --env "PYTHONPATH=$OUTPUT/sim_deps:$ISAAC_DEPS:$PROJECT/scripts:$COLLECTOR"
 --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?Slurm must assign one GPU}"
 --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y --env TERM=xterm --env OMP_NUM_THREADS=4
 --pwd "$PROJECT" "$SIF_FILE" /opt/IsaacLab/isaaclab.sh -p)
endpoint="tcp://127.0.0.1:$((22000 + SLURM_JOB_ID % 20000))"
worker_pid=
cleanup() { if [[ -n "$worker_pid" ]]; then kill "$worker_pid" 2>/dev/null || true; wait "$worker_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
"$PYTHON" -u scripts/tracking_online_policy.py --coordinate-velocity "$CHECKPOINT" --endpoint "$endpoint" >"$OUTPUT/policy.log" 2>&1 &
worker_pid=$!
"${container[@]}" "$PROJECT/scripts/tracking_online_sim.py" --collector "$COLLECTOR" --manifest "$OUTPUT/manifest.json" \
 --output "$OUTPUT" --endpoint "$endpoint" --seconds 10 >"$OUTPUT/simulator.log" 2>&1
cleanup
worker_pid=
"$PYTHON" -u scripts/summarize_tracking_online_three.py --output "$OUTPUT" --no-videos >"$OUTPUT/summary.log" 2>&1
"$PYTHON" -u scripts/merge_coordinate_velocity_eval.py --previous "$PREVIOUS" --new "$OUTPUT" \
 --output "$COMBINED" >"$ROOT/merge.log" 2>&1
"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);s=json.loads((root/'comparison_with_existing/summary.json').read_text())
assert s['complete'] and s['pairing_valid'] and s['recorded_rollouts']==150
assert len(s['comparison_videos'])==30 and all(v['frames']==250 and len(v['model_order'])==5 for v in s['comparison_videos'])
index=root.parents[1]
for name,target in [('tracking_comparison_with_velocity.png',root/'comparison_with_existing/tracking_comparison.png'),
                    ('comparison_videos_with_velocity',root/'comparison_with_existing/comparison_videos'),
                    ('latest_with_velocity',root/'comparison_with_existing')]:
 p=index/name
 assert not p.exists() or p.is_symlink(),f'Refusing to replace existing regular file: {p}'
 temp=index/(name+'.tmp');temp.symlink_to(target);temp.replace(p)
(root/'README.md').write_text('Only Coordinate+Velocity was evaluated in this run.\n\n'
 'live_tracking/: 30 new raw closed-loop trials and all three camera views.\n'
 'comparison_with_existing/: fifth-method plots, 30 five-panel videos, and 150-trial summary including 120 saved trials.\n'
 'provenance/: validation-selected checkpoint and frozen evaluation sources.\n')
print('COORDINATE+VELOCITY COMPARISON COMPLETE:',root)
PY
