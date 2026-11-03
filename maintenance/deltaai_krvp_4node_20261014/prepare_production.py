"""Check frozen approval receipts; do not repeat profiling or the dataset audit."""
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['DVLA_PROJECT_DIR'])
from core.dom_training import load_family
from utils.dom_experiment import atomic_json,digest,file_digest

HERE=Path(__file__).resolve().parent
ROOT=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/krvp-scale-ddp-20261014')
def verified(path):
    v=json.loads(path.read_text());assert digest({k:x for k,x in v.items() if k!='fingerprint'})==v['fingerprint'];return v
def main():
    manifest=load_family(ROOT);selected=verified(ROOT/'selected-runtime.json');recipe=verified(ROOT/'recipe.json')
    assert selected['world_size']==int(os.environ['SLURM_JOB_NUM_NODES'])*4==16
    assert selected['variant']=='krvp100'
    assert selected['manifest']==recipe['manifest']==manifest['fingerprint']
    assert recipe['runtime_fingerprint']==digest(selected['runtime'])
    profiles=[verified(p) for p in (ROOT/'distributed-profiles').glob('*.json')]
    profile=next(p for p in profiles if p['fingerprint']==selected['distributed_profile_fingerprint'])
    assert profile['variant']=='krvp100' and profile['future_inputs_ignored']
    assert profile['world_size']==selected['world_size'] and profile['benchmark_samples']==70400 and profile['measured_updates']==180
    assert all(profile.get(k) for k in ['complete','safe','strict_reload','rank_rng_reload','full_data_benchmark','sustained_reload_passed','post_training_validation_passed'])
    for name,expected in manifest['artifacts'].items():assert file_digest(ROOT/name)==expected,name
    reuse=json.loads((ROOT/'provenance/verification-reuse.json').read_text());report=json.loads((HERE/'verified.json').read_text())
    assert file_digest(HERE/'verified.json')==reuse['derived_receipt_sha256']
    assert report['manifest']==manifest['fingerprint'] and report['state']=='complete' and not report['errors'] and report['completed_files']==804900
    job=os.environ['SLURM_JOB_ID']
    old=json.loads((HERE/'status.json').read_text()) if (HERE/'status.json').exists() else {}
    old.update(stage='production_starting',training_job=job,training_started=False,production_held=False,
        family=str(ROOT),source=os.environ['DVLA_PROJECT_DIR'],selected_runtime=selected,
        manifest_fingerprint=manifest['fingerprint'],recipe_fingerprint=recipe['fingerprint'],
        verification_complete=True,verification_reused=True,verified_files=804900,
        readiness_complete=True,readiness_reused=True,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic_json(HERE/'status.json',old)

if __name__=='__main__':main()
