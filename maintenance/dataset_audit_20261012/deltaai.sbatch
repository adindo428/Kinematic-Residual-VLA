#!/bin/bash -l
#SBATCH --job-name=dom-audit
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/dataset-audit-20261012/audit-%A_%a.out
#SBATCH --error=/work/nvme/bivv/aparameshwaran/dynamicvla/dataset-audit-20261012/audit-%A_%a.err
set -euo pipefail
module load python/miniforge3_pytorch/2.12.0
source /work/nvme/bivv/aparameshwaran/dynamicvla/venv-krvla-dom-20261008/bin/activate
export LD_LIBRARY_PATH="/work/nvme/bivv/aparameshwaran/dynamicvla/native-krvla-dom/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -z "${AUDIT_LIMIT:-}" ]]; then
python - <<'VERIFY_SMOKE'
import json
p='/work/nvme/bivv/aparameshwaran/dynamicvla/dataset-audit-20261012/smoke2/status-00.json'
r=json.load(open(p))
assert r['state']=='complete' and r['completed_episodes']==8 and r['failed_episodes']==0 and not r['metadata_errors'], r
VERIFY_SMOKE
fi
python /work/nvme/bivv/aparameshwaran/dynamicvla/dataset-audit-20261012/audit.py --family /work/nvme/bivv/aparameshwaran/dynamicvla/domv-video-clean-ddp-20261012 --output /work/nvme/bivv/aparameshwaran/dynamicvla/dataset-audit-20261012/${AUDIT_OUTPUT:-results} --shard "${SLURM_ARRAY_TASK_ID}" --shards 1 --workers 32 ${AUDIT_LIMIT:+--limit "$AUDIT_LIMIT"}
