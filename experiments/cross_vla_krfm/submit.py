"""Run on Palmetto. Idempotent job DAG with receipts saved after each sbatch."""
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'submissions.json'
state=json.loads(STATE.read_text())
setup=state['setup']['job_id']

def persist():
    t=STATE.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');os.replace(t,STATE)

def submit(name,command,dependency):
    key=name+'_'+command
    if key in state:return state[key]['job_id']
    script=ROOT/f'{key}.sbatch'
    expected_name=name+'-'+command[:4]
    # Reconcile a previously uncertain scheduler response rather than blindly
    # submitting again. Command path also has to identify this campaign.
    lines=subprocess.check_output(['squeue','-u','aparame','-h','-n',expected_name,'-o','%i'],text=True).split()
    matching=[]
    for job in lines:
        details=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        if f'Command={script}' in details:matching.append(job)
    if len(matching)>1:raise RuntimeError('Multiple existing matching jobs; manual reconciliation needed')
    if matching:
        job=matching[0]
    else:
        result=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+dependency,str(script)],text=True,capture_output=True)
        if result.returncode:raise RuntimeError(result.stderr)
        job=result.stdout.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('Uncertain scheduler response; do not retry without reconciliation')
    state[key]={'job_id':job,'recipe':name,'command':command,'dependency':dependency,'script':str(script),'gpus':2,'gpu':'H200'}
    persist();print(key,job,flush=True)
    return job

for recipe in sorted((ROOT/'recipes').glob('*.json')):
    name=recipe.stem
    profile=submit(name,'profile',setup)
    submit(name,'train',profile)
