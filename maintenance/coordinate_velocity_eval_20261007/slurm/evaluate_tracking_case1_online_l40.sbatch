#!/usr/bin/env bash
#SBATCH --job-name=tracking-online-stage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=l40:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err
set -euo pipefail
WORKSPACE=/home/aparame/Research/Dynamic_VLA
PROJECT=${PROJECT:-$WORKSPACE/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA}
COLLECTOR=${COLLECTOR:-$PROJECT/../Data_Collection_Dynamic_Objects}
PYTHON=${PYTHON:-/home/aparame/.conda/envs/dynamicvla/bin/python}
ROOT75=/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/checkpoints
BASELINE=${BASELINE:-$ROOT75/tracking_dom75_full20_baseline_16623336}
RESIDUAL=${RESIDUAL:-$ROOT75/tracking_dom75_full20_residual_16623337}
COORDINATE=${COORDINATE:-$ROOT75/tracking_dom75_full20_coordinate_16623338}
MODE=${MODE:-smoke}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORKSPACE/RESULTS/tracking_current_eval/$MODE}
OUTPUT=$OUTPUT_ROOT/${RUN_ID:-${SLURM_JOB_ID:?Set RUN_ID outside Slurm}}
DATASET_ROOT=${DATASET_ROOT:-/scratch/aparame/.cache/huggingface/lerobot/adi2440/dom-franka-tracking}
SCENE_DIR=${SCENE_DIR:-/scratch/aparame/Dynamic_VLA/assets/DOM-3D-Scenes}
OBJECT_DIR=${OBJECT_DIR:-/scratch/aparame/Dynamic_VLA/dom_benchmark/objects}
SIF_FILE=${SIF_FILE:-/scratch/aparame/isaaclab.sif}
ISAAC_DEPS=${ISAAC_DEPS:-/scratch/aparame/Dynamic_VLA/isaac_eval_deps/python}
export HF_HOME=${HF_HOME:-/scratch/aparame/.cache/huggingface}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HF_HOME/lerobot}
export LD_LIBRARY_PATH=/software/spackages/linux-rocky8-x86_64/gcc-9.5.0/cuda-12.1.1-d4ik3tbvtk3ypske36p4ogeulsnsv26p/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
unset LEROBOT_HOME
cd "$PROJECT"
case "$MODE" in full|smoke|policy-smoke|preflight) ;; *) echo "Invalid MODE=$MODE" >&2; exit 2 ;; esac
# Each invocation gets a fresh output directory; previous results are never overwritten.
if [[ -e "$OUTPUT" ]]; then echo "Output already exists: $OUTPUT" >&2; exit 2; fi
mkdir -p "$OUTPUT"
coordinate_args=(--coordinate "$COORDINATE")
if [[ -n "${VELOCITY:-}" ]]; then coordinate_args+=(--velocity "$VELOCITY"); fi
max_episodes=0; seconds=10
if [[ "$MODE" == smoke ]]; then max_episodes=1; seconds=2; fi
prepare=("$PYTHON" scripts/prepare_tracking_online.py --dataset "$DATASET_ROOT" --scene-dir "$SCENE_DIR" --object-dir "$OBJECT_DIR"
 --collector "$COLLECTOR" --baseline "$BASELINE" --residual "$RESIDUAL" "${coordinate_args[@]}" --output "$OUTPUT/manifest.json" --max-episodes "$max_episodes")
if [[ "${REQUIRE_COMPLETE_TRAINING:-1}" == 1 ]]; then prepare+=(--require-complete-training); fi
if [[ -n "${ROBOT_USD:-}" ]]; then prepare+=(--robot-usd "$ROBOT_USD"); fi
"${prepare[@]}"
if [[ "$MODE" == preflight ]]; then exit 0; fi
if [[ "$MODE" == policy-smoke ]]; then
 "$PYTHON" -u scripts/tracking_online_policy.py --smoke --dataset "$DATASET_ROOT" --baseline "$BASELINE" --residual "$RESIDUAL" "${coordinate_args[@]}" --output "$OUTPUT"
 exit 0
fi
command -v apptainer >/dev/null
[[ -f "$SIF_FILE" ]]
mkdir -p "$OUTPUT/kit_data" "$OUTPUT/kit_cache" "$OUTPUT/sim_tmp" "$OUTPUT/sim_deps"
container=(apptainer exec --cleanenv --nv -B "$WORKSPACE:$WORKSPACE" -B /scratch/aparame:/scratch/aparame
 -B "$OUTPUT/kit_data:/isaac-sim/kit/data" -B "$OUTPUT/kit_cache:/isaac-sim/kit/cache" -B "$OUTPUT/sim_tmp:/tmp"
 --env "PYTHONPATH=$OUTPUT/sim_deps:$ISAAC_DEPS:$PROJECT/scripts:$COLLECTOR"
 --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?Slurm must assign a GPU}"
 --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y --env TERM=xterm --env OMP_NUM_THREADS=4 --pwd "$PROJECT" "$SIF_FILE" /opt/IsaacLab/isaaclab.sh -p)
"${container[@]}" -m pip install --disable-pip-version-check --no-deps --target "$OUTPUT/sim_deps" \
 'av==16.0.1' 'pyarrow==19.0.1' 'Pillow==11.3.0' 'PyYAML==6.0.2' 'pyzmq==26.4.0' >"$OUTPUT/dependencies.log" 2>&1
endpoint="tcp://127.0.0.1:${POLICY_PORT:-$((22000 + SLURM_JOB_ID % 20000))}"
worker_pid=
cleanup() { if [[ -n "$worker_pid" ]]; then kill "$worker_pid" 2>/dev/null || true; wait "$worker_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
"$PYTHON" -u scripts/tracking_online_policy.py --baseline "$BASELINE" --residual "$RESIDUAL" "${coordinate_args[@]}" --endpoint "$endpoint" >"$OUTPUT/policy.log" 2>&1 &
worker_pid=$!
set +e
"${container[@]}" "$PROJECT/scripts/tracking_online_sim.py" --collector "$COLLECTOR" --manifest "$OUTPUT/manifest.json" \
 --output "$OUTPUT" --endpoint "$endpoint" --seconds "$seconds" >"$OUTPUT/simulator.log" 2>&1
sim_status=$?
set -e
cleanup
worker_pid=
if [[ -f "$OUTPUT/rollouts.json" ]]; then
 "$PYTHON" scripts/summarize_tracking_online_three.py --output "$OUTPUT" >"$OUTPUT/summary.log" 2>&1
fi
if (( sim_status != 0 )); then echo "Simulation failed; see $OUTPUT/simulator.log" >&2; exit "$sim_status"; fi
[[ -f "$OUTPUT/completion.json" ]] || { echo 'Simulator exited without completion marker' >&2; exit 1; }
echo "Completed live evaluation: $OUTPUT/summary.json"
