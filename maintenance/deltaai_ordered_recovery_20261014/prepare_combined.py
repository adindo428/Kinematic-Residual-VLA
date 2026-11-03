"""One bounded check, with fresh processes for checkpoint smoke and full-data work."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from coordinate import HERE,FAMILY,SOURCE,state,verified,candidates,select,atomic_json,digest,file_digest


def main():
    lock=(HERE/'combined-preparation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=os.environ['SLURM_JOB_ID'];manifest=verified(FAMILY/'manifest.json')
    verification=json.loads((HERE/'verified.json').read_text());reuse=json.loads((FAMILY/'provenance/verification-reuse.json').read_text())
    assert file_digest(HERE/'verified.json')==reuse['derived_receipt_sha256']
    assert verification['manifest']==manifest['fingerprint'] and verification['state']=='complete' and not verification['errors']
    assert verification['completed_files']==verification['total_files']==804900
    assert verification['counts']=={'video':402448,'parquet':201224,'phase':201224,'metadata':4}
    state('isolated_readiness_starting',training_job=job,combined_allocation=True,production_submitted=True,
        training_started=False,error=None,verification_complete=True,verification_reused=True,
        verified_files=804900,verification_job='3370672',family=str(FAMILY),source=str(SOURCE),dependency=None)
    from core.dom_training import load_family
    load_family(FAMILY)
    selected=FAMILY/'selected-runtime.json'
    if selected.exists():
        chosen=verified(selected);recipe=verified(FAMILY/'recipe.json')
        assert chosen['manifest']==recipe['manifest']==manifest['fingerprint']
        assert recipe['runtime_fingerprint']==digest(chosen['runtime'])
        state('training_starting',training_job=job,selected_runtime=chosen,recipe_fingerprint=recipe['fingerprint'],readiness_reused=True)
        return
    assert not (FAMILY/'runs/domv100/provenance.json').exists()
    paths=candidates();assert len(paths)==1
    runtime=verified(Path(paths[0]))['runtime'];assert runtime['microbatch']==22 and runtime['workers']==1
    state('isolated_checkpoint_smoke',profile_job=job,candidate_files=paths,readiness_total_limit_minutes=15)
    deadline=time.monotonic()+15*60
    def run(phase,command,env):
        remaining=int(deadline-time.monotonic());assert remaining>0,'Readiness time limit reached'
        output=FAMILY/'logs'/('isolated-'+phase+'-'+job+'.log')
        with output.open('ab',buffering=0) as stream:
            subprocess.run(['timeout','--signal=TERM','--kill-after=30s',str(remaining)+'s']+command,
                cwd=str(SOURCE),env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    env=dict(os.environ,WANDB_MODE='disabled',DOM100_RUNTIME_FILE=paths[0],DOM100_COMMAND='smoke',PYTHONFAULTHANDLER='1')
    run('checkpoint',['bash','scripts/run_dom100_distributed.sh','domv100'],env)
    state('isolated_full_data_check',profile_job=job)
    # Full-data benchmark starts with a fresh Python/CUDA/DDP process set.
    run('full-data',[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',
        str(FAMILY/'native_benchmark.py'),'--family-root',str(FAMILY),'--runtime-file',paths[0]],env)
    chosen,recipe=select()
    atomic_json(HERE/('combined-prepared-'+job+'.json'),{'job_id':job,'manifest':manifest['fingerprint'],
        'verified_files':804900,'verification_job':'3370672','runtime_fingerprint':chosen['fingerprint'],
        'recipe_fingerprint':recipe['fingerprint'],'readiness_seconds':15*60-(deadline-time.monotonic())})
    state('training_starting',training_job=job,selected_runtime=chosen,recipe_fingerprint=recipe['fingerprint'],readiness_complete=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('isolated_readiness_failed',error=type(exc).__name__+': '+str(exc),training_started=False)
        raise
