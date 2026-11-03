#!/bin/bash -l
#SBATCH --job-name=krvp-opt
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:2
#SBATCH --constraint=gpu_a100_40gb
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/aparame/Dynamic_VLA/phase-optimization-20261013/opt-%j.out
#SBATCH --error=/scratch/aparame/Dynamic_VLA/phase-optimization-20261013/opt-%j.err
set -euo pipefail
export DVLA_PROJECT_DIR=/scratch/aparame/Dynamic_VLA/third_party/dom100-phase-two-gpu-20261013
export DOM100_ROOT=/scratch/aparame/Dynamic_VLA/dom100-phase-two-gpu-20261013
cd "$DVLA_PROJECT_DIR"
source scripts/dom100_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
out=/scratch/aparame/Dynamic_VLA/phase-optimization-20261013/results-${SLURM_JOB_ID}
mkdir -p "$out"
bench=/scratch/aparame/Dynamic_VLA/phase-optimization-20261013/benchmark.py
for spec in 22:1 22:4 44:4; do
 mb=${spec%:*}; workers=${spec#*:}
 python -m torch.distributed.run --standalone --nproc-per-node=2 "$bench" --root "$DOM100_ROOT" --out "$out/mb${mb}-w${workers}.json" --mb "$mb" --workers "$workers" > "$out/mb${mb}-w${workers}.log" 2>&1 || echo "Candidate $spec failed; inspect its log"
done
mb=$(python - "$out" <<'PY'
import json,sys
from pathlib import Path
rows=[json.loads(p.read_text()) for p in Path(sys.argv[1]).glob('mb*-w4.json')]
rows=[r for r in rows if r['safe']]
if not rows:raise SystemExit('No safe four-worker candidate; do not change production')
print(max(rows,key=lambda r:r['samples_per_second'])['runtime']['microbatch'])
PY
)
for sampling in frames blocks; do
 python -m torch.distributed.run --standalone --nproc-per-node=2 "$bench" --root "$DOM100_ROOT" --out "$out/pilot-${sampling}.json" --mb "$mb" --workers 4 --sampling "$sampling" --steps 2000 --validate > "$out/pilot-${sampling}.log" 2>&1
done
python - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);a=json.loads((p/'pilot-frames.json').read_text());b=json.loads((p/'pilot-blocks.json').read_text())
r={'frames':a,'blocks':b,'eligible_for_phase_run':bool(a['safe'] and b['safe'] and b['samples_per_second']>=1.1*a['samples_per_second'] and b['validation']['h20/action_mse']<=1.05*a['validation']['h20/action_mse']),'production_modified':False,'note':'Phase-run paired pilot only; does not approve block sampling for other architectures. Resume transition requires explicit provenance and cursor handling.'}
(p/'decision.json').write_text(json.dumps(r,indent=2)+'\n')
PY
