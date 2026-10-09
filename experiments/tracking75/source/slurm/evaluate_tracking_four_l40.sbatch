#!/usr/bin/env bash
#SBATCH --job-name=tracking-four-eval
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=l40:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err
set -euo pipefail
export PROJECT=/home/aparame/Research/Dynamic_VLA/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA
cd "$PROJECT"
export HF_HOME=/scratch/aparame/.cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_LEROBOT_HOME=$HF_HOME/lerobot
export LD_LIBRARY_PATH=/software/spackages/linux-rocky8-x86_64/gcc-9.5.0/cuda-12.1.1-d4ik3tbvtk3ypske36p4ogeulsnsv26p/lib64:${LD_LIBRARY_PATH:-}
unset LEROBOT_HOME
PYTHON=/home/aparame/.conda/envs/dynamicvla/bin/python
export BASELINE=${BASELINE:-/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/checkpoints/tracking_dom75_full20_baseline_16623336}
export RESIDUAL=${RESIDUAL:-/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/checkpoints/tracking_dom75_full20_residual_16623337}
export COORDINATE=${COORDINATE:-/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/resume_outputs/checkpoints/tracking_dom75_full20_coordinate_16623338}
export VELOCITY=${VELOCITY:?Set VELOCITY to the completed DOM+Velocity checkpoint}
stage=${EVAL_STAGE:-all}
case "$stage" in all|offline|live) ;; *) echo "Invalid EVAL_STAGE=$stage" >&2; exit 2;; esac
root=${COMPARISON_ROOT:-/home/aparame/Research/Dynamic_VLA/RESULTS/tracking_comparison_four_75epoch}/${COMPARISON_RUN_ID:-$SLURM_JOB_ID}
if [[ "$stage" != live ]]; then
[[ ! -e "$root" ]] || { echo "Refusing to overwrite $root" >&2; exit 2; }
mkdir -p "$root/provenance"
cp slurm/README_tracking_four.md "$root/README.md"
for variant in baseline velocity residual coordinate; do
 case "$variant" in baseline) ckpt=$BASELINE;; velocity) ckpt=$VELOCITY;; residual) ckpt=$RESIDUAL;; coordinate) ckpt=$COORDINATE;; esac
 "$PYTHON" -u scripts/verify_tracking_training.py --variant "$variant" --checkpoint "$ckpt"
 cp "$ckpt/training_ready.json" "$root/provenance/${variant}_training_ready.json"
done
MODE=policy-smoke OUTPUT_ROOT="$root/verification/policy" bash slurm/evaluate_tracking_case1_online_l40.sbatch
"$PYTHON" -u scripts/evaluate_tracking_four_offline.py --baseline "$BASELINE" --velocity "$VELOCITY" --residual "$RESIDUAL" --coordinate "$COORDINATE" --output "$root/offline_imitation"
if [[ "$stage" == offline ]]; then echo "A100 OFFLINE EVALUATION COMPLETE: $root"; exit 0; fi
else
 [[ -f "$root/offline_imitation/comparison.json" ]] || { echo "Missing completed offline evaluation: $root" >&2; exit 2; }
fi
MODE=smoke OUTPUT_ROOT="$root/verification/live" bash slurm/evaluate_tracking_case1_online_l40.sbatch
"$PYTHON" - "$root/verification/live/$SLURM_JOB_ID/summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert s['complete'] and s['pairing_valid'] and s['recorded_rollouts']==12
assert len(s['comparison_videos'])==3 and all(v['frames']==50 and len(v['model_order'])==4 for v in s['comparison_videos'])
PY
# Put live artifacts directly beside offline_imitation, matching the retained layout.
MODE=full OUTPUT_ROOT="$root" RUN_ID=live_tracking bash slurm/evaluate_tracking_case1_online_l40.sbatch
"$PYTHON" - "$root/live_tracking/summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert s['complete'] and s['pairing_valid'] and s['recorded_rollouts']==120
assert len(s['comparison_videos'])==30 and all(len(v['model_order'])==4 for v in s['comparison_videos'])
print('FOUR-METHOD COMPARISON COMPLETE:',sys.argv[1])
PY
