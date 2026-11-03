"""Submit production once, after the successful actual four-GPU interactive check."""
import fcntl,json,subprocess
from pathlib import Path
from coordinate import HERE,FAMILY,SOURCE,verified,atomic_json,file_digest,state

def main():
    lock=(HERE/'production-launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=HERE/'production-submission.json'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if value.get('job_id'):print(json.dumps(value));return
        raise RuntimeError('Inspect uncertain submission before retry')
    check=json.loads((HERE/'interactive-check-3381007.json').read_text())
    assert check['passed'] and not check['training_started']
    manifest=verified(FAMILY/'manifest.json');chosen=verified(FAMILY/'selected-runtime.json')
    recipe=verified(FAMILY/'recipe.json')
    profiles=[verified(p) for p in (FAMILY/'distributed-profiles').glob('*.json')]
    profile=next(p for p in profiles if p['fingerprint']==chosen['distributed_profile_fingerprint'])
    assert profile['benchmark_job_id']=='3381007'
    assert all(profile[k] for k in ('complete','safe','full_data_benchmark','strict_reload','rank_rng_reload','policy_before_validation_prefetch'))
    assert profile['benchmark_samples']==6336 and profile['measured_updates']==16
    assert manifest['fingerprint']==chosen['manifest']==recipe['manifest']
    from core.dom_training import load_family
    load_family(FAMILY)
    old=subprocess.check_output(['scontrol','show','job','3379435'],universal_newlines=True)
    assert 'JobState=PENDING' in old and 'Reason=JobHeldUser' in old
    batch=HERE/'combined.sbatch'
    assert str(FAMILY) in batch.read_text() and str(HERE) in batch.read_text()
    value={'stage':'submitting','batch':str(batch),'batch_sha256':file_digest(batch),
        'interactive_check_job':'3381007','interactive_check_passed':True,'dependency':None,
        'family':str(FAMILY),'manifest':manifest['fingerprint'],'recipe':recipe['fingerprint'],
        'selected_runtime':chosen,'superseded_held_job':'3379435'}
    atomic_json(receipt,value)
    result=subprocess.run(['sbatch','--parsable',str(batch)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    value.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    if result.returncode==0 and result.stdout.strip().split(';')[0].isdigit():
        value.update(stage='submitted',job_id=result.stdout.strip().split(';')[0])
    atomic_json(receipt,value)
    if not value.get('job_id'):raise RuntimeError('Submission outcome requires inspection')
    # Cancel only the obsolete, unstarted held submission after new submission is recorded.
    subprocess.check_call(['scancel','3379435'])
    value['old_held_job_cancelled']=True;atomic_json(receipt,value)
    state('production_submitted',training_job=value['job_id'],production_held=False,
        production_submitted=True,training_started=False,readiness_complete=True,readiness_reused=True,
        interactive_check_job='3381007',readiness_candidate_limit_minutes=15,
        selected_runtime=chosen,recipe_fingerprint=recipe['fingerprint'],
        old_training_job='3379435',old_training_cancelled=True)
    print(json.dumps(value),flush=True)

if __name__=='__main__':main()
