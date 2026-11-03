"""Submit the authorized one-allocation readiness check and conditional restart once."""
import fcntl,json,subprocess,time,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100')
from utils.dom_experiment import atomic_json,file_digest

def main():
    lock=(HERE/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=HERE/'combined-submission.json'
    if p.exists():
        v=json.loads(p.read_text())
        if v.get('job_id'):print(json.dumps(v));return
        raise RuntimeError('Inspect uncertain submission before retry')
    deployment=json.loads((HERE/'deployment.json').read_text())
    batch=HERE/'combined.sbatch'
    v={'stage':'submitting','batch':str(batch),'batch_sha256':file_digest(batch),'dependency':None,
       'purpose':'15minute isolated check; conditional training in same48hour allocation',
       'incident_job':'3370672','requested_exclusion':'gh055; diagnostic choice, not a confirmed node fault'}
    atomic_json(p,v)
    r=subprocess.run(['sbatch','--parsable',str(batch)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    v.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    if r.returncode==0 and r.stdout.strip().split(';')[0].isdigit():
        v.update(job_id=r.stdout.strip().split(';')[0],stage='submitted')
    atomic_json(p,v)
    if not v.get('job_id'):raise RuntimeError('Submission failed: '+r.stderr)
    state={'stage':'isolated_check_and_training_queued','training_job':v['job_id'],
        'production_submitted':True,'training_started':False,'combined_allocation':True,'dependency':None,
        'family':deployment['family'],'source':deployment['source'],'manifest_fingerprint':deployment['manifest'],
        'verification_complete':True,'verification_reused':True,'verified_files':804900,'verification_job':'3370672',
        'readiness_candidate_limit_minutes':15,'readiness_total_limit_minutes':15,
        'previous_failed_job':'3370672','candidate_microbatch':22,'candidate_workers':1,
        'native_cause_confirmed':False,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    atomic_json(HERE/'status.json',state);print(json.dumps(v),flush=True)

if __name__=='__main__':main()
