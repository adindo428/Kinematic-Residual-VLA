#!/usr/bin/env bash
# Read-only status for the authorized full-data continuation on DeltaAI.
set -euo pipefail
krv_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
krv_local="$krv_repo/scratch/dom45/training/krv100-full-deltaai-20261102"
python3 - "$krv_local" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
for name in ('jobs.json','upload-status.json','release-status.json'):
    if (r/name).exists():
        print(name+': '+json.dumps(json.loads((r/name).read_text())))
PY
ssh -o ControlPath=/tmp/deltaai-kinematic.sock -o BatchMode=yes \
  anayak2@gh-login01.delta.ncsa.illinois.edu /usr/bin/python3 - <<'PY'
import json,subprocess
from pathlib import Path
r=Path('/work/nvme/bicn/anayak2/dynamicvla/krv100-full-resume-epoch002-20261102')
subprocess.run(['squeue','-u','anayak2','-o','%.12i %.20j %.12P %.12T %.12M %R'])
for name in ('migration.json','verification-progress.json','verified-data.json','interactive-readiness.json','readiness.json',
             'training-status.json','family/runs/krv100/latest.json',
             'family/runs/krv100/training_complete.json'):
    p=r/name
    if p.exists():
        value=json.loads(p.read_text())
        if name in ('readiness.json','interactive-readiness.json'):
            value={k:v for k,v in value.items() if k not in ('records',)}
        print(name+': '+json.dumps(value),flush=True)
logs=sorted([p for pattern in ('train-*.err','preflight-*.err','smoke-*.err')
             for p in (r/'logs').glob(pattern)],key=lambda p:p.stat().st_mtime)
if logs:
    print('\nLast log: '+str(logs[-1]),flush=True)
    subprocess.run(['tail','-n','12',str(logs[-1])])
PY
