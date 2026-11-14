"""Release held H200 jobs only after the CPU-routed copy and preflight pass."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import psutil
import palmetto_arai_allocated_transport as transport

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'scratch/dom45/training/full-distance-arai-h2004-20261113'
REMOTE='/scratch/arai/Dynamic_VLA/full-distance-h2004-20261113'

def status(phase,**details):
    path=STAGE/'launch-status.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**details),indent=2)+'\n');temp.replace(path)

def main():
    jobs=json.loads((STAGE/'jobs.json').read_text())
    assert set(jobs)=={'smolvla_kr','pi05_kr','vlash_kr'}
    assert all(j['gpu']=='H200' and j['gpus']==4 for j in jobs.values())
    status('held_waiting_for_full_dataset',jobs=jobs)
    def data_complete():
        try:return json.loads((STAGE/'data-transfer/complete.json').read_text()).get('complete') is True
        except (FileNotFoundError,json.JSONDecodeError):return False
    while not data_complete():
        pid=int((STAGE/'data-transfer/coordinator.pid').read_text())
        try:running=psutil.Process(pid).is_running() and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:running=False
        if not running:
            time.sleep(1)
            if data_complete():break
            raise RuntimeError('Dataset copy stopped before completion; GPU jobs remain held')
        time.sleep(30)
    status('held_running_cpu_preflight',jobs=jobs)
    command='source '+shlex.quote(REMOTE+'/env.sh')+'; export CUDA_VISIBLE_DEVICES=""; '+shlex.join([REMOTE+'/env/bin/python','-u',REMOTE+'/cpu_prepare.py'])+' > '+shlex.quote(REMOTE+'/cpu-preparation.log')+' 2>&1'
    p=subprocess.run(transport.allocated_command(['bash','-lc',command]),capture_output=True,text=True,timeout=1200)
    (STAGE/'cpu-preflight-transport.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise RuntimeError('Full-data CPU preflight failed; GPU jobs remain held. Inspect remote cpu-preparation.log')
    code='''import json,os,hashlib,subprocess
from pathlib import Path
r=Path(REMOTE)
assert os.environ['SLURM_JOB_ID']=='20264351' and 'login' not in os.environ['SLURMD_NODENAME']
assert json.loads((r/'cpu-readiness.json').read_text())['passed']
prepared=json.loads((r/'prepared-palmetto.json').read_text())
for name,expected in prepared['files'].items():
 h=hashlib.sha256((r/name).read_bytes()).hexdigest();assert h==expected,name
jobs=json.loads((r/'jobs.json').read_text())
for model,j in jobs.items():
 assert j['gpus']==4 and j['gpu']=='H200'
 description=subprocess.check_output(['scontrol','show','job',str(j['job_id']),'-o'],universal_newlines=True)
 assert 'JobState=PENDING' in description and 'Reason=JobHeldUser' in description,description
for j in jobs.values():subprocess.check_call(['scontrol','release',str(j['job_id'])])
ids=','.join(str(j['job_id']) for j in jobs.values())
print(subprocess.check_output(['squeue','-j',ids,'-o','%.18i %.24j %.10T %.6D %R'],universal_newlines=True))
receipt={'released':True,'cpu_control_job':os.environ['SLURM_JOB_ID'],'jobs':jobs,'cpu_readiness':json.loads((r/'cpu-readiness.json').read_text())}
(r/'release-receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print('RELEASE_JSON',json.dumps(receipt))
'''.replace('REMOTE',repr(REMOTE))
    p=subprocess.run(transport.allocated_command(['/usr/bin/python3','-c',code]),capture_output=True,text=True,timeout=120)
    (STAGE/'release-receipt.txt').write_text(p.stdout+p.stderr)
    if p.returncode:raise RuntimeError('CPU-node release failed; inspect release-receipt.txt')
    for line in p.stdout.splitlines():
        if line.startswith('RELEASE_JSON '):
            receipt=json.loads(line[len('RELEASE_JSON '):]);(STAGE/'release-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    status('submitted_released_after_full_data_validation',jobs=jobs)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status('launch_failed_gpu_jobs_not_released',error=str(exc));raise
