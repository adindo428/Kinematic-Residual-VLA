set -eu
DOMV_JOB=$(python3 -c 'import json; print(json.load(open("/scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114/submission.json"))["slurm_job_id"])')
squeue -j "$DOMV_JOB" -o '%.16i %.22j %.10T %.30R'
python3 - <<'PY'
from pathlib import Path
import json
root=Path('/scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114')
job=json.loads((root/'submission.json').read_text())['slurm_job_id']
for name in ['submission.json','disposable-check/startup.json','disposable-check/preflight.json','startup.json','domv/status.json','domv/best_checkpoint.json']:
    path=root/name
    print(name, path.read_text() if path.exists() else 'Not available yet')
for suffix in ['out','err']:
    path=root/f'logs/train-{job}.{suffix}'
    if path.exists():print(path.name,'\n'.join(path.read_text().splitlines()[-20:]))
PY
