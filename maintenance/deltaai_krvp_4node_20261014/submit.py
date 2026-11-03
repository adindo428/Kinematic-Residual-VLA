"""Idempotent submission; never resubmit automatically after an unknown outcome."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/krvp-scale-ddp-20261014')
SOURCE=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100')
sys.path.insert(0,str(SOURCE))
from utils.dom_experiment import atomic_json,digest,file_digest,source_digest

def main():
    lock=(HERE/'submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=HERE/'production-submission.json'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if value.get('job_id'):print(json.dumps(value),flush=True);return
        raise RuntimeError('Inspect uncertain submission; do not submit again blindly')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert digest({k:v for k,v in manifest.items() if k!='fingerprint'})==manifest['fingerprint']
    assert manifest['run_variant']=='krvp100' and source_digest(SOURCE)==manifest['source_sha256']
    for name,expected in manifest['artifacts'].items():assert file_digest(ROOT/name)==expected,name
    batch=HERE/'production.sbatch';assert file_digest(batch)==manifest['artifacts']['production.sbatch']
    assert '#SBATCH --nodes=4' in batch.read_text() and '#SBATCH --gpus-per-node=4' in batch.read_text()
    assert not (ROOT/'runs/krvp100/provenance.json').exists()
    value={'stage':'submitting','batch':str(batch),'batch_sha256':file_digest(batch),'family':str(ROOT),
        'variant':'krvp100','nodes':4,'world_size':16,'dependency':None,'manifest':manifest['fingerprint'],
        'readiness':'One15-minute bounded16-GPU check, then training in same allocation',
        'submitted_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    atomic_json(receipt,value)
    result=subprocess.run(['sbatch','--parsable',str(batch)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    value.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    if result.returncode==0 and result.stdout.strip().split(';')[0].isdigit():
        value.update(job_id=result.stdout.strip().split(';')[0],stage='submitted')
    atomic_json(receipt,value)
    if not value.get('job_id'):raise RuntimeError('Submission outcome requires inspection')
    status=json.loads((HERE/'status.json').read_text())
    status.update(stage='production_submitted_pending_readiness',training_job=value['job_id'],
        production_submitted=True,readiness_complete=False,training_started=False,
        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic_json(HERE/'status.json',status)
    print(json.dumps(value),flush=True)

if __name__=='__main__':main()
