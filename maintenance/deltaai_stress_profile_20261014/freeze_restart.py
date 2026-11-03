"""Freeze runtime recovery separately, preserving failed source/family and verified bytes."""
import json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORK=Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
OLD=WORK/'domv-audited-20261012'
ROOT=WORK/'domv-stress-ddp-20261014'
SOURCE=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100')
PREVIOUS=WORK/'setup-existing-data-domv-20261012'
sys.path.insert(0,str(SOURCE))
from utils.dom_experiment import atomic_json,digest,file_digest,source_digest

def main():
    assert not ROOT.exists(),'Restart family already exists; inspect before retry'
    old=json.loads((OLD/'manifest.json').read_text());assert digest({k:v for k,v in old.items() if k!='fingerprint'})==old['fingerprint']
    assert old['source_sha256']==source_digest(SOURCE)=='f991015a229476400e0fc772177af33759d1bc3f23901381e2f83c7faaeab921'
    for name,h in old['artifacts'].items():assert file_digest(OLD/name)==h,name
    report=json.loads((PREVIOUS/'verified.json').read_text())
    receipt=json.loads((PREVIOUS/'combined-verification-receipt.json').read_text())
    assert file_digest(PREVIOUS/'verified.json')==receipt['sha256']
    assert report['manifest']==old['fingerprint'] and report['state']=='complete' and not report['errors']
    assert report['completed_files']==report['total_files']==804900
    ROOT.mkdir();(ROOT/'provenance').mkdir();(ROOT/'logs').mkdir()
    for name in old['artifacts']:
        p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to((OLD/name).resolve())
    for name in ['phase_cache','initialization']:
        if not (ROOT/name).exists():(ROOT/name).symlink_to((OLD/name).resolve(),target_is_directory=True)
    if not (ROOT/'initialization/model.safetensors').exists():
        (ROOT/'initialization/model.safetensors').symlink_to((OLD/'initialization/model.safetensors').resolve())
    assert (ROOT/'initialization/model.safetensors').is_file()
    (ROOT/'native_benchmark.py').write_bytes((HERE/'native_benchmark.py').read_bytes())
    manifest=dict(old);manifest.pop('fingerprint')
    manifest.update(initialization=str(ROOT/'initialization'),runtime_adaptation_parent=old['fingerprint'],
        runtime_recovery={'incident_job':'3370672','cause':'Rank0 native SIGSEGV; exact component not confirmed',
            'checkpoint_and_benchmark':'Separate fresh CUDA/DDP processes','candidate_microbatches':[22,44,88],
            'candidate_workers':[2,4],'global_batch':352,'benchmark_driver':'native_benchmark.py',
            'model_optimizer_sampler_code':'unchanged','dataset_split_normalization':'unchanged'})
    manifest['artifacts']=dict(old['artifacts'],**{'native_benchmark.py':file_digest(ROOT/'native_benchmark.py')})
    manifest['fingerprint']=digest(manifest);atomic_json(ROOT/'manifest.json',manifest)
    recipe=json.loads((OLD/'recipe.json').read_text());assert digest({k:v for k,v in recipe.items() if k!='fingerprint'})==recipe['fingerprint']
    recipe.pop('fingerprint');recipe['manifest']=manifest['fingerprint'];recipe.pop('runtime_fingerprint',None)
    recipe['fingerprint']=digest(recipe);atomic_json(ROOT/'recipe.json',recipe)
    # A derived reuse receipt records the new manifest without claiming a new scan.
    derived=dict(report);derived.update(manifest=manifest['fingerprint'],verification_reused=True,
        original_verified_manifest=old['fingerprint'],original_verified_receipt_sha256=receipt['sha256'])
    atomic_json(HERE/'verified.json',derived)
    atomic_json(ROOT/'provenance/verification-reuse.json',{'original_receipt_sha256':receipt['sha256'],
        'derived_receipt_sha256':file_digest(HERE/'verified.json'),'original_report':report,
        'original_manifest':old['fingerprint'],'same_dataset_root':old['dataset_root'],
        'all_parent_artifact_hashes_retained':True,'phase_cache':str((ROOT/'phase_cache').resolve()),
        'read_only_cache':True,'new_scan_performed':False})
    atomic_json(ROOT/'provenance/parent-manifest.json',old)
    (HERE/'bounded-single-gpu.json').write_bytes((PREVIOUS/'bounded-single-gpu.json').read_bytes())
    deployment={'family':str(ROOT),'source':str(SOURCE),'source_sha256':manifest['source_sha256'],
        'manifest':manifest['fingerprint'],'recipe':recipe['fingerprint'],'split_sha256':manifest['split_sha256'],
        'verified_files_reused':804900,'previous_failed_job':'3370672'}
    atomic_json(HERE/'deployment.json',deployment);print(json.dumps(deployment),flush=True)

if __name__=='__main__':main()
