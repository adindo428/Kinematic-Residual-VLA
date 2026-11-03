"""Submit approved full training once; preserve uncertain outcomes for inspection."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['DVLA_PROJECT_DIR'])
from utils.dom_experiment import atomic_json,file_digest,digest
from approve_runtime import HERE,ROOT,verified
def main():
    lock=(HERE/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=HERE/'production-submission.json'
    if p.exists():
        old=json.loads(p.read_text())
        if old.get('job_id'):print(json.dumps(old));return
        raise RuntimeError('Inspect uncertain submission before retry')
    chosen=verified(ROOT/'selected-runtime.json');recipe=verified(ROOT/'recipe.json');manifest=verified(ROOT/'manifest.json')
    assert chosen['world_size']==8 and chosen['node_count']==2 and chosen['gpus_per_node']==4
    assert chosen['manifest']==recipe['manifest']==manifest['fingerprint']
    assert recipe['runtime_fingerprint']==digest(chosen['runtime'])
    assert not (ROOT/'runs/domv100/provenance.json').exists()
    old=subprocess.check_output(['scontrol','show','job','3381061'],universal_newlines=True)
    assert 'JobState=PENDING' in old and 'Reason=JobHeldUser' in old
    batch=HERE/'production.sbatch'
    assert '#SBATCH --nodes=2' in batch.read_text() and str(ROOT) in batch.read_text()
    value={'stage':'submitting','batch':str(batch),'batch_sha256':file_digest(batch),'dependency':None,
        'profile_job':os.environ['SLURM_JOB_ID'],'family':str(ROOT),'manifest':manifest['fingerprint'],
        'recipe':recipe['fingerprint'],'selected_runtime':chosen,'superseded_held_job':'3381061'}
    atomic_json(p,value)
    result=subprocess.run(['sbatch','--parsable',str(batch)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    value.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    if result.returncode==0 and result.stdout.strip().split(';')[0].isdigit():value.update(job_id=result.stdout.strip().split(';')[0],stage='submitted')
    atomic_json(p,value)
    if not value.get('job_id'):raise RuntimeError('Submission outcome requires inspection')
    subprocess.check_call(['scancel','3381061']);value['old_held_job_cancelled']=True;atomic_json(p,value)
    atomic_json(HERE/'status.json',{'stage':'production_submitted','training_job':value['job_id'],
        'profile_job':value['profile_job'],'interactive_check_job':value['profile_job'],
        'production_submitted':True,'production_held':False,'training_started':False,'family':str(ROOT),
        'source':os.environ['DVLA_PROJECT_DIR'],'manifest_fingerprint':manifest['fingerprint'],
        'recipe_fingerprint':recipe['fingerprint'],'selected_runtime':chosen,'dependency':None,
        'readiness_complete':True,'readiness_reused':True,'verification_complete':True,'verification_reused':True,
        'verified_files':804900,'verification_job':'3370672','old_training_job':'3381061','old_training_cancelled':True,
        'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    print(json.dumps(value),flush=True)
if __name__=='__main__':main()
