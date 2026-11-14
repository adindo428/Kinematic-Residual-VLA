#!/usr/bin/env bash
# Live Delta scheduler status; shared metadata via the healthy GH login client.
set -euo pipefail
ssh -o ControlPath=/tmp/delta-gpu-aparameshwaran.sock -o BatchMode=yes \
 aparameshwaran@login.delta.ncsa.illinois.edu \
 'squeue -u aparameshwaran -o "%.18i %.24P %.24j %.12T %.8M %.6D %.16b %R"'
ssh -o ControlPath=/tmp/deltaai-kinematic.sock -o BatchMode=yes aparameshwaran@gh-login /usr/bin/python3 - <<'PY'
from pathlib import Path
import json,subprocess
r=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/dom-full-matched-delta-20261107')
for name in ('jobs.json','prepared.json','environment-check.json','auth-check.json','interactive-passed.json',
 'profile-world2.json','profile-world16.json','training-status.json','family/runs/dom100/best.json',
 'family/runs/dom100/latest.json','evaluation-ready.json'):
 p=r/name
 if p.exists():
  x=json.loads(p.read_text())
  if name.startswith('profile-world'):x={k:v for k,v in x.items() if k not in ('hardware_by_rank','validation_metrics')}
  print(name+': '+json.dumps(x),flush=True)
logs=sorted((r/'logs').glob('*.err'),key=lambda p:p.stat().st_mtime)
if logs:
 print('Latest log: '+str(logs[-1]),flush=True);subprocess.run(['tail','-n','15',str(logs[-1])])
PY
