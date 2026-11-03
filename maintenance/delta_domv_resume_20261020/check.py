"""One bounded hardware/resume check in the same allocation as production."""
import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['DVLA_PROJECT_DIR'])
from utils.dom_experiment import atomic_json,digest,file_digest
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ['DOM100_ROOT'])

def main():
    deployment=json.loads((HERE/'deployment.json').read_text())
    for name,h in deployment['files'].items():assert file_digest(HERE/name)==h,name
    if (HERE/'approval.json').exists():
        approval=json.loads((HERE/'approval.json').read_text())
        assert approval['deployment']==deployment['fingerprint']
        return
    job=os.environ['SLURM_JOB_ID'];deadline=time.monotonic()+15*60
    receipt={'job_id':job,'passed':False,'complete':False,'world_size':8,'cluster':'Delta','variant':'domv100'}
    path=HERE/('check-'+job+'.json');atomic_json(path,receipt)
    env=dict(os.environ,WANDB_MODE='disabled',DOM100_RUNTIME_FILE=str(HERE/'runtime.json'))
    try:
        for phase in ['smoke','benchmark']:
            log=HERE/'logs'/(phase+'-'+job+'.log')
            with log.open('ab',buffering=0) as stream:
                result=subprocess.run(['timeout','--signal=TERM','--kill-after=30s',str(max(1,int(deadline-time.monotonic())))+'s',
                    'bash',str(HERE/'launch_ranks.sh'),phase],env=env,stdout=stream,stderr=subprocess.STDOUT)
            receipt[phase+'_exit']=result.returncode;receipt[phase+'_log']=str(log);atomic_json(path,receipt)
            if result.returncode:raise RuntimeError(phase+' failed: '+str(result.returncode))
        resume=json.loads((HERE/('resume-check-'+job+'.json')).read_text())
        assert resume['passed'] and resume['world_size']==8
        assert resume['checkpoint_fingerprint']==deployment['checkpoint_fingerprint']
        profiles=[json.loads(p.read_text()) for p in (ROOT/'distributed-profiles').glob('*.json')]
        profile=next(p for p in profiles if p.get('benchmark_job_id')==job and p.get('benchmark_driver_sha256')==deployment['files']['native_benchmark.py'])
        assert digest({k:v for k,v in profile.items() if k!='fingerprint'})==profile['fingerprint']
        assert profile['world_size']==8 and profile['benchmark_samples']==70400 and profile['measured_updates']==180
        assert all(profile.get(k) for k in ['complete','safe','strict_reload','rank_rng_reload','full_data_benchmark','sustained_reload_passed','post_training_validation_passed'])
        approval={'deployment':deployment['fingerprint'],'profile_fingerprint':profile['fingerprint'],
            'source_manifest':deployment['manifest'],'checkpoint_fingerprint':resume['checkpoint_fingerprint'],
            'world_size':8,'hardware':profile['hardware'],'runtime':profile['runtime'],
            'profile_job':job,'samples_per_second':profile['samples_per_second'],'real_resume_passed':True}
        atomic_json(HERE/'approval.json',approval)
        receipt.update(passed=True,complete=True,profile=profile,resume=resume)
        atomic_json(path,receipt)
        status=json.loads((HERE/'status.json').read_text());status.update(stage='production_starting',readiness_complete=True,approval=approval)
        atomic_json(HERE/'status.json',status)
    except BaseException as exc:
        receipt.update(complete=True,error=str(exc));atomic_json(path,receipt)
        status=json.loads((HERE/'status.json').read_text());status.update(stage='readiness_failed',error=str(exc),readiness_complete=False)
        atomic_json(HERE/'status.json',status)
        raise

if __name__=='__main__':main()
