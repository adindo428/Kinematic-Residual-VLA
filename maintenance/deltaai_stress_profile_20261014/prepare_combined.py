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
    raise RuntimeError('Runtime must be selected by completed interactive stress sweep before production')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('isolated_readiness_failed',error=type(exc).__name__+': '+str(exc),training_started=False)
        raise
