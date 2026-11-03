#!/usr/bin/env bash
# Hardware alternatives share the existing production writer lock and W&B run.
set -euo pipefail
source scripts/dom100_env.sh
variant=${1:-krv100}
set +e
python - "${DOM100_ROOT}/runs/${variant}" "${SLURM_JOB_ID:-local}" <<'PY'
import fcntl
import json
from pathlib import Path
import sys
from utils.dom_experiment import atomic_json

root, job = Path(sys.argv[1]), sys.argv[2]
if (root/'training_complete.json').exists():
    atomic_json(root/f'allocation-{job}.json',{'job_id':job,'outcome':'skipped_completed_run'})
    print('Shared experiment is already complete; releasing this allocation.',flush=True)
    sys.exit(3)
root.mkdir(parents=True,exist_ok=True)
with (root/'.lock').open('a') as lock:
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        atomic_json(root/f'allocation-{job}.json',{'job_id':job,'outcome':'skipped_active_writer'})
        print('Another allocation is training this experiment; releasing this allocation.',flush=True)
        sys.exit(3)
# The check releases its probe lock. The trainer acquires the same lock for
# its entire run, resolving races if two allocations verify simultaneously.
PY
status=$?
set -e
if (( status == 3 )); then exit 0; fi
if (( status != 0 )); then exit "${status}"; fi
exec bash scripts/run_dom100_distributed.sh "${variant}"
