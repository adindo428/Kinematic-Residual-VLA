"""Submit the authorized seven-hour continuation exactly once."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/work/nvme/bivv/aparameshwaran/dynamicvla/training-sources/delta-domv-f991015a')
from utils.dom_experiment import atomic_json,digest,file_digest

def main():
    lock=(HERE/'submit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=HERE/'submission.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('job_id'):print(json.dumps(old));return
        raise RuntimeError('Inspect uncertain submission before retry')
    deployment=json.loads((HERE/'deployment.json').read_text())
    assert digest({k:v for k,v in deployment.items() if k!='fingerprint'})==deployment['fingerprint']
    for name,h in deployment['files'].items():assert file_digest(HERE/name)==h,name
    value={'stage':'submitting','cluster':'Delta','account':'bivv-delta-gpu','deployment':deployment['fingerprint'],
        'allocation_hours':7,'maximum_charge_units':168,'world_size':8,'dependency':None,
        'submitted_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    atomic_json(path,value)
    result=subprocess.run(['sbatch','--parsable',str(HERE/'production.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    value.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    if result.returncode==0 and result.stdout.strip().split(';')[0].isdigit():value.update(stage='submitted',job_id=result.stdout.strip().split(';')[0])
    atomic_json(path,value)
    if not value.get('job_id'):raise RuntimeError('Submission outcome requires inspection')
    atomic_json(HERE/'status.json',dict(value,stage='queued_pending_readiness',operational_epoch_cap=10,
        family=deployment['family'],source=deployment['source'],variant='domv100',readiness_complete=False,
        checkpoint_progress=deployment['initial_progress'],checkpoint_fingerprint=deployment['checkpoint_fingerprint']))
    print(json.dumps(value),flush=True)

if __name__=='__main__':main()
