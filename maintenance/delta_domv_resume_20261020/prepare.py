"""Verify the shared frozen experiment and scope its ten-epoch control to Delta."""
import argparse,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,os.environ['DVLA_PROJECT_DIR'])
from core.dom_training import load_family
from utils.dom_experiment import atomic_json,digest,file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpointed',action='store_true');args=p.parse_args()
    root=Path(os.environ['DOM100_ROOT']);deployment=json.loads((HERE/'deployment.json').read_text())
    assert digest({k:v for k,v in deployment.items() if k!='fingerprint'})==deployment['fingerprint']
    for name,h in deployment['files'].items():assert file_digest(HERE/name)==h,name
    manifest=load_family(root);assert manifest['fingerprint']==deployment['manifest']
    recipe=json.loads((root/'recipe.json').read_text());assert recipe['fingerprint']==deployment['recipe']
    latest=json.loads((root/'runs/domv100/latest.json').read_text())
    assert latest['fingerprint']==deployment['checkpoint_fingerprint']
    assert latest['progress']['updates']>=deployment['initial_progress']['updates']
    job=os.environ['SLURM_JOB_ID']
    assert int(os.environ['SLURM_JOB_NUM_NODES'])==1
    control={'version':1,'epochs':10,'scheduler_horizon_epochs':100,
        'authorization':'User requested DOM+velocity resume on newly allocated NCSA Delta GPU resources, retaining ten total epochs and existing optimization trajectory',
        'source_sha256':manifest['source_sha256'],
        'trainer_path':str(Path(os.environ['DVLA_PROJECT_DIR'])/'core/dom_distributed_training.py'),
        'trainer_sha256':file_digest(Path(os.environ['DVLA_PROJECT_DIR'])/'core/dom_distributed_training.py'),
        'bootstrap_sha256':deployment['files']['sitecustomize.py'],
        'jobs':{job:{'family':str(root),'variant':'domv100','manifest':manifest['fingerprint'],'world_size':8}}}
    control['fingerprint']=digest(control);atomic_json(HERE/'control.json',control)
    old=json.loads((HERE/'status.json').read_text()) if (HERE/'status.json').exists() else {}
    old.update(stage='checkpointed_allocation_end' if args.checkpointed else ('production_starting' if (HERE/'approval.json').exists() else 'bounded_h200_readiness'),
        job_id=job,cluster='Delta',account='bivv-delta-gpu',family=str(root),variant='domv100',world_size=8,
        source=os.environ['DVLA_PROJECT_DIR'],operational_epoch_cap=10,scheduler_horizon_epochs=100,
        checkpoint_progress=latest['progress'],checkpoint_fingerprint=latest['fingerprint'],
        control_file=str(HERE/'control.json'),control_fingerprint=control['fingerprint'],
        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        allocation_hours=7,maximum_allocation_charge=168,charge_factor=3,readiness_limit_minutes=15)
    atomic_json(HERE/'status.json',old)
    atomic_json(root/'provenance/delta-resume-20261020.json',dict(old,deployment_fingerprint=deployment['fingerprint']))

if __name__=='__main__':main()
