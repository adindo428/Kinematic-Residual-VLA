#!/usr/bin/env bash
# One user-authorized interactive comparison; no retries or production commands.
set -euo pipefail
cd /work/nvme/bivv/aparameshwaran/dynamicvla/pi05-comparison-20261021
[[ ! -e interactive-requested.json && ! -e allocation-id ]]
[[ $(tail -n 1 import-preflight-3.log) == 'Pinned pi05 imports passed' ]]
grep -q '^OK$' cpu-contracts.log
grep -q 'coordinated stop: PASS' collective-contract.log
grep -q 'Verified train and validation two-camera decoding' data-preflight.log
env/bin/python - <<'PY'
import json,datetime,pathlib
p=pathlib.Path('interactive-requested.json')
with p.open('x') as f:json.dump({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'waiting_for_verified_checkpoint','partition':'ghx4-interactive','gpus':2,'cpus':32,'memory_gib':256,'max_hours':2,'production_changed':False},f,indent=2)
PY
# stage.py renames only after matching the immutable source SHA256.
while [[ ! -f checkpoint.pt ]]; do sleep 10; done
[[ $(stat -c %s checkpoint.pt) == 40456772509 ]]
# Refuse a duplicate even if another process submitted after staging.
[[ -z $(squeue -h -u "$USER" -n pi05-compare-20261021 -o %i) ]]
salloc -A bivv-dtai-gh -p ghx4-interactive --nodes=1 --ntasks=1 \
  --gpus-per-node=2 --cpus-per-task=32 --mem=256G --time=02:00:00 \
  --immediate=60 --job-name=pi05-compare-20261021 \
  bash -c 'printf "%s\n" "$SLURM_JOB_ID" > allocation-id; exec srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-task=2 --cpu-bind=cores bash sweep.sh' \
  > interactive-allocation.log 2>&1
