"""Bounded, isolated four-GPU stress sweep; never starts production."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from coordinate import HERE,FAMILY,SOURCE,verified,candidates,select,atomic_json,state,digest

def main():
    lock=(HERE/'stress-check.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=os.environ['SLURM_JOB_ID'];manifest=verified(FAMILY/'manifest.json')
    assert not (FAMILY/'runs/domv100/provenance.json').exists()
    state('interactive_stress_running',training_job='3381061',production_held=True,
        production_submitted=True,training_started=False,interactive_check_job=job,
        family=str(FAMILY),source=str(SOURCE),manifest_fingerprint=manifest['fingerprint'])
    deadline=time.monotonic()+35*60;trials=[]
    paths=candidates()
    for path in paths:
        if deadline-time.monotonic()<240:break
        runtime=verified(Path(path))['runtime'];trial={'runtime':runtime,'runtime_file':path,'passed':False}
        trial_deadline=min(deadline,time.monotonic()+9*60)
        prefix=Path(path).stem
        state('interactive_stress_candidate',stress_runtime=runtime,interactive_check_job=job)
        env=dict(os.environ,WANDB_MODE='disabled',DOM100_RUNTIME_FILE=path,DOM100_COMMAND='smoke',PYTHONFAULTHANDLER='1')
        def run(phase,command):
            remaining=int(trial_deadline-time.monotonic());assert remaining>0,'Candidate time limit reached'
            output=FAMILY/'logs'/('stress-'+phase+'-'+job+'-'+prefix+'.log')
            trial[phase+'_log']=str(output)
            with output.open('ab',buffering=0) as stream:
                result=subprocess.run(['timeout','--signal=TERM','--kill-after=30s',str(remaining)+'s']+command,
                    cwd=str(SOURCE),env=env,stdout=stream,stderr=subprocess.STDOUT)
            trial[phase+'_exit']=result.returncode
            if result.returncode:raise RuntimeError(phase+' exit '+str(result.returncode))
        started=time.monotonic()
        try:
            run('checkpoint',['bash','scripts/run_dom100_distributed.sh','domv100'])
            run('full-data',[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',
                str(FAMILY/'native_benchmark.py'),'--family-root',str(FAMILY),'--runtime-file',path])
            profiles=[verified(p) for p in (FAMILY/'distributed-profiles').glob('*.json')]
            matches=[p for p in profiles if p.get('full_data_benchmark') and p.get('post_training_validation_passed')
                and p.get('benchmark_job_id')==job and p['runtime']==runtime and p['manifest']==manifest['fingerprint']]
            assert len(matches)==1 and matches[0]['safe']
            trial.update(passed=True,profile=matches[0])
        except Exception as exc:trial['error']=type(exc).__name__+': '+str(exc)
        trial['seconds']=time.monotonic()-started;trials.append(trial)
        atomic_json(HERE/('stress-check-'+job+'.json'),{'interactive_job':job,'complete':False,'trials':trials})
        print(json.dumps({'runtime':runtime,'passed':trial['passed'],'error':trial.get('error'),'seconds':trial['seconds']}),flush=True)
    approved=[t for t in trials if t['passed']]
    result={'interactive_job':job,'complete':True,'trials':trials,'any_passed':bool(approved),'training_started':False}
    if approved:
        chosen,recipe=select();result.update(selected_runtime=chosen,recipe_fingerprint=recipe['fingerprint'])
        state('interactive_stress_passed',selected_runtime=chosen,recipe_fingerprint=recipe['fingerprint'],
            readiness_complete=True,training_started=False,production_held=True,interactive_check_job=job)
    else:state('interactive_stress_failed',readiness_complete=False,training_started=False,production_held=True)
    atomic_json(HERE/('stress-check-'+job+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='trials'}),flush=True)

if __name__=='__main__':main()
