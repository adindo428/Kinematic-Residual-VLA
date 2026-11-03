"""Verify bytes and select an actually measured runtime inside the training allocation."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from coordinate import HERE,FAMILY,SOURCE,atomic_json,candidates,digest,file_digest,select,state,verified


def verification_ready(manifest):
    path=HERE/'verified.json'
    if not path.exists():return None
    report=json.loads(path.read_text())
    expected={'video':402448,'parquet':201224,'phase':201224,'metadata':4}
    if not (report['state']=='complete' and not report['errors']
            and report['manifest']==manifest['fingerprint']
            and report['split_sha256']==manifest['split_sha256']
            and report['completed_files']==report['total_files']==804900
            and report['counts']==expected):
        raise ValueError('Incomplete or incompatible byte-verification receipt')
    receipt=HERE/'combined-verification-receipt.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['sha256']!=file_digest(path):raise ValueError('Byte-verification receipt changed')
    else:
        atomic_json(receipt,{'sha256':file_digest(path),'manifest':manifest['fingerprint'],
            'verified_in_job':report['job_id'],'cache_read_only':True})
    return report


def runtime_ready(manifest,hardware):
    path=FAMILY/'selected-runtime.json'
    if not path.exists():return None
    chosen=verified(path);recipe=verified(FAMILY/'recipe.json')
    key=digest({'hardware':hardware,'world_size':4,'runtime':chosen['runtime'],'variant':'domv100'})[:16]
    profile=verified(FAMILY/'distributed-profiles'/(key+'.json'))
    if not (chosen['manifest']==recipe['manifest']==profile['manifest']==manifest['fingerprint']
            and chosen['world_size']==profile['world_size']==4
            and chosen['global_effective_batch']==recipe['effective_batch']==352
            and recipe['runtime_fingerprint']==digest(chosen['runtime'])
            and chosen['distributed_profile_fingerprint']==profile['fingerprint']
            and profile['variant']=='domv100'
            and profile['complete'] and profile['safe'] and profile['full_data_benchmark']
            and profile['strict_reload'] and profile['rank_rng_reload']):
        raise ValueError('Runtime approval does not match this allocation and final split')
    return chosen,recipe


def main():
    lock=(HERE/'combined-preparation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=os.environ['SLURM_JOB_ID']
    state('combined_preparation_starting',training_job=job,production_submitted=True,
          combined_allocation=True,dependency=None,family=str(FAMILY),source=str(SOURCE))
    manifest=verified(FAMILY/'manifest.json')
    # Requeues reuse the completed byte scan only for this frozen, read-only cache.
    report=verification_ready(manifest)
    if report is None:
        state('verifying_existing_dataset',verification_job=job,verification_reused=False)
        subprocess.run([sys.executable,str(HERE/'verify_existing.py')],check=True,timeout=3600)
        report=verification_ready(manifest)
    else:state('existing_dataset_verification_reused',verification_reused=True)
    state('existing_dataset_verified',verification_job=report['job_id'],
          verified_files=report['completed_files'],verification_complete=True)

    import torch
    import lerobot
    from core.dom_training import load_family
    assert torch.cuda.device_count()==4
    assert all('GH200' in torch.cuda.get_device_name(i) for i in range(4))
    load_family(FAMILY)
    download=json.loads((Path(os.environ['DVLA_WORK_ROOT'])/'setup-20261008/hzxie--DOM-status.json').read_text())
    assert download['state']=='complete' and download['revision']==manifest['dataset_revision']
    props=torch.cuda.get_device_properties(0)
    hardware={'name':props.name,'total_memory':props.total_memory,'torch':torch.__version__,
              'cuda':torch.version.cuda,'capability':[props.major,props.minor]}
    ready=runtime_ready(manifest,hardware)
    if ready is None:
        if (FAMILY/'runs/domv100/provenance.json').exists():
            raise ValueError('Existing training provenance requires its original approved runtime')
        paths=candidates()
        assert len(paths)==1,'Exactly one measured readiness candidate is permitted'
        state('four_gpu_readiness',profile_job=job,candidate_files=paths,
              readiness_candidate_limit_minutes=15,readiness_total_limit_minutes=15)
        env=dict(os.environ,WANDB_MODE='disabled',DOM100_COMMAND='profile',DOM100_RUNTIME_FILE=paths[0])
        log=FAMILY/'logs'/('combined-readiness-'+job+'.log')
        with log.open('ab',buffering=0) as stream:
            subprocess.run(['timeout','--signal=TERM','--kill-after=60s','15m',
                'bash','scripts/run_dom100_distributed.sh','domv100'],cwd=str(SOURCE),env=env,
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        select()
        ready=runtime_ready(manifest,hardware)
        assert ready is not None
    chosen,recipe=ready
    atomic_json(HERE/('combined-prepared-'+job+'.json'),{'job_id':job,'manifest':manifest['fingerprint'],
        'verified_files':report['completed_files'],'verification_job':report['job_id'],
        'runtime_fingerprint':chosen['fingerprint'],'recipe_fingerprint':recipe['fingerprint']})
    state('training_starting',selected_runtime=chosen,recipe_fingerprint=recipe['fingerprint'],
          readiness_complete=True,production_submitted=True,training_job=job)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('combined_preparation_failed',error=type(exc).__name__+': '+str(exc),training_started=False)
        raise
