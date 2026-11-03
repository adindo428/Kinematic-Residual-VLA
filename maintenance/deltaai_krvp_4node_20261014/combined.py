"""One bounded 16-rank readiness attempt, then training in this allocation."""
import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['DVLA_PROJECT_DIR'])
from utils.dom_experiment import atomic_json,digest,file_digest
from core.dom_training import load_family

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ['DOM100_ROOT'])

def status(stage,**fields):
    path=HERE/'status.json'
    value=json.loads(path.read_text()) if path.exists() else {}
    value.update(stage=stage,training_job=os.environ['SLURM_JOB_ID'],variant='krvp100',
        training_started=False,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields)
    atomic_json(path,value)

def main():
    manifest=load_family(ROOT)
    for name in ['combined.py','scale_check.py','approve_runtime.py','prepare_production.py','production.sbatch']:
        assert file_digest(HERE/name)==manifest['artifacts'][name],name
    assert int(os.environ['SLURM_JOB_NUM_NODES'])==4
    if (ROOT/'selected-runtime.json').exists():
        # A continuation must retain the exact approved recipe and all rank RNGs.
        subprocess.check_call([sys.executable,str(HERE/'prepare_production.py')])
        status('production_starting',readiness_complete=True,readiness_reused=True)
        return
    assert not (ROOT/'runs/krvp100/provenance.json').exists()
    status('bounded_16gpu_readiness',readiness_complete=False,readiness_reused=False,
        readiness_total_limit_minutes=15,verification_complete=True,verification_reused=True,verified_files=804900)
    env=dict(os.environ,WANDB_MODE='disabled')
    result=subprocess.run(['timeout','--signal=TERM','--kill-after=30s','16m',
        sys.executable,str(HERE/'scale_check.py')],env=env)
    if result.returncode:
        status('readiness_failed',readiness_complete=False,readiness_exit=result.returncode)
        raise SystemExit(result.returncode)
    try:
        subprocess.check_call([sys.executable,str(HERE/'approve_runtime.py')],env=env)
        subprocess.check_call([sys.executable,str(HERE/'prepare_production.py')])
    except Exception as exc:
        status('readiness_approval_failed',readiness_complete=False,error=str(exc))
        raise
    status('production_starting',readiness_complete=True,readiness_reused=False,
        profile_job=os.environ['SLURM_JOB_ID'])

if __name__=='__main__':main()
