#!/usr/bin/env bash
#SBATCH --job-name=tracking-table1
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=l40s:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%A_%a.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%A_%a.err
set -euo pipefail
WORKSPACE=/home/aparame/Research/Dynamic_VLA
STAGE=/scratch/aparame/Dynamic_VLA/runs/tracking_table1_20261008
PROJECT=$STAGE/source
COLLECTOR=$STAGE/collector
PYTHON=/home/aparame/.conda/envs/dynamicvla/bin/python
ROOT=${ROOT:?Set ROOT to this benchmark run}
MODE=${MODE:?Set MODE to pilot or full}
SIF_FILE=/scratch/aparame/isaaclab.sif
ISAAC_DEPS=/scratch/aparame/Dynamic_VLA/isaac_eval_deps/python
export HF_HOME=/scratch/aparame/.cache/huggingface HF_HUB_OFFLINE=1
export HF_LEROBOT_HOME=$HF_HOME/lerobot
export LD_LIBRARY_PATH=/software/spackages/linux-rocky8-x86_64/gcc-9.5.0/cuda-12.1.1-d4ik3tbvtk3ypske36p4ogeulsnsv26p/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
unset LEROBOT_HOME
cd "$PROJECT"
command -v apptainer >/dev/null
scratch_run=$ROOT/runtime/${MODE}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$scratch_run/kit_data" "$scratch_run/kit_cache" "$scratch_run/sim_tmp"
if [[ "$MODE" == pilot ]]; then
 mkdir "$STAGE/sim_deps"
fi
container=(apptainer exec --cleanenv --nv -B "$WORKSPACE:$WORKSPACE" -B /scratch/aparame:/scratch/aparame
 -B "$scratch_run/kit_data:/isaac-sim/kit/data" -B "$scratch_run/kit_cache:/isaac-sim/kit/cache" -B "$scratch_run/sim_tmp:/tmp"
 --env "PYTHONPATH=$STAGE/sim_deps:$ISAAC_DEPS:$PROJECT/scripts:$COLLECTOR"
 --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?Slurm must assign one GPU}"
 --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y --env TERM=xterm --env OMP_NUM_THREADS=4
 --pwd "$PROJECT" "$SIF_FILE" /opt/IsaacLab/isaaclab.sh -p)
if [[ "$MODE" == pilot ]]; then
 "${container[@]}" -m pip install --disable-pip-version-check --no-deps --target "$STAGE/sim_deps" \
 'av==16.0.1' 'pyarrow==19.0.1' 'Pillow==11.3.0' 'PyYAML==6.0.2' 'pyzmq==26.4.0' >"$ROOT/dependencies.log" 2>&1
else
 [[ -f "$ROOT/pilot_verified.json" ]]
fi
mapfile -t checkpoints < <("$PYTHON" - "$ROOT/protocol.json" <<'PY'
import json,sys
for k,v in json.load(open(sys.argv[1]))['models'].items():print('--'+k.replace('_','-'));print(v)
PY
)
if [[ "$MODE" == pilot ]]; then
 "$PYTHON" -u scripts/tracking_online_policy.py --smoke "${checkpoints[@]}" --output "$ROOT/provenance" >"$ROOT/policy_smoke.log" 2>&1
 shards=(0 10 20 30 40 50 60 70 80 61 62 63 64)
else
 shards=("${SLURM_ARRAY_TASK_ID:?}")
fi
endpoint="tcp://127.0.0.1:$((22000 + SLURM_JOB_ID % 20000))"
worker_pid=
cleanup() { if [[ -n "$worker_pid" ]]; then kill "$worker_pid" 2>/dev/null || true; wait "$worker_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
"$PYTHON" -u scripts/tracking_online_policy.py "${checkpoints[@]}" --endpoint "$endpoint" >"$scratch_run/policy.log" 2>&1 &
worker_pid=$!
for shard in "${shards[@]}"; do
 printf -v shard_name '%03d' "$shard"
 OUTPUT=$ROOT/$MODE/$shard_name
 [[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite $OUTPUT" >&2; exit 2; }
 mkdir -p "$OUTPUT"
 "$PYTHON" - "$ROOT/shards/$shard_name.json" "$OUTPUT/manifest.json" "$MODE" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
if sys.argv[3]=='pilot':m['episodes']=m['episodes'][:1]
json.dump(m,open(sys.argv[2],'w'),indent=2)
PY
 seconds=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["seconds"])' "$OUTPUT/manifest.json")
 video_args=(); if [[ "$MODE" == pilot ]]; then video_args=(--all-videos); fi
 "${container[@]}" "$PROJECT/scripts/tracking_online_sim.py" --collector "$COLLECTOR" --manifest "$OUTPUT/manifest.json" \
 --output "$OUTPUT" --endpoint "$endpoint" --seconds "$seconds" "${video_args[@]}" >"$OUTPUT/simulator.log" 2>&1
 "$PYTHON" scripts/verify_benchmark.py --manifest "$OUTPUT/manifest.json" --output "$OUTPUT" >"$OUTPUT/verification.log" 2>&1
 # Each simulator shuts down its policy worker; restart for the next pilot shard.
 wait "$worker_pid"; worker_pid=
 if [[ "$shard" != "${shards[-1]}" ]]; then
  "$PYTHON" -u scripts/tracking_online_policy.py "${checkpoints[@]}" --endpoint "$endpoint" >>"$scratch_run/policy.log" 2>&1 &
  worker_pid=$!
 fi
done
if [[ "$MODE" == pilot ]]; then
 "$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);p=json.load(open(r/'protocol.json'))
reports=[json.load(open(r/'pilot'/f'{s:03d}'/'verification.json')) for s in p['pilot_shards']]
assert len(reports)==13 and all(v['verified'] and v['pairing_valid'] and v['rollouts']==4 for v in reports)
json.dump({'verified':True,'shards':13,'rollouts':52,'policy_smoke':True},open(r/'pilot_verified.json','w'),indent=2)
PY
fi
