from pathlib import Path
import json,hashlib,subprocess
from datetime import datetime,timezone
H=Path(__file__).resolve().parent;R='/scratch/aparame/Dynamic_VLA/checkpoint-cleanup-20261026';P='/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python'
SSH=['ssh','-F','/dev/null','-S','/private/tmp/vlcm-palmetto-k1vtw3i7/socket','-o','BatchMode=yes','-o','ConnectTimeout=5','aparame@slogin.palmetto.clemson.edu']
def call(s,data=None):return subprocess.run(SSH+[s],input=data,capture_output=True,check=True,timeout=30).stdout.decode()
assert not (H/'submission.json').exists(),'Recorded/uncertain submission; reconcile first'
assert 'checkpoint-cleanup-oct26' not in call('squeue -u aparame -h -o "%j"')
call('test ! -e '+R+' && mkdir -p '+R)
for f in ('cleanup.py','plan.json'):call('cat > '+R+'/'+f,(H/f).read_bytes())
script='''#!/usr/bin/env bash
#SBATCH --job-name=checkpoint-cleanup-oct26
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output='''+R+'''/cleanup-%j.out
#SBATCH --error='''+R+'''/cleanup-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd '''+R+'\n'+P+' -u cleanup.py\n'
(H/'cleanup.sbatch').write_text(script);call('cat > '+R+'/cleanup.sbatch',script.encode())
pre=call('bash -n '+R+'/cleanup.sbatch && sbatch --test-only '+R+'/cleanup.sbatch 2>&1');(H/'preflight.txt').write_text(pre)
state={'state':'intent','root':R,'files':41,'bytes':806490103923,'scope':'Exact older/superseded checkpoints from plan.json; user requested older copies removed on2026-10-26','at':datetime.now(timezone.utc).isoformat(),'script_sha256':hashlib.sha256(script.encode()).hexdigest(),'plan_sha256':hashlib.sha256((H/'plan.json').read_bytes()).hexdigest()};(H/'submission.json').write_text(json.dumps(state,indent=2)+'\n');call('cat > '+R+'/submission.json',json.dumps(state).encode())
raw=call('sbatch --parsable '+R+'/cleanup.sbatch').strip();job=raw.split(';')[0];assert job.isdigit(),raw
state.update(state='submitted',job_id=job);(H/'submission.json').write_text(json.dumps(state,indent=2)+'\n');call('cat > '+R+'/submission.json',json.dumps(state).encode());print(json.dumps(state))
