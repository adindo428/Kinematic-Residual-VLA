"""Freeze a separate phase-gate run while reusing the approved data audit."""
import json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORK=Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
OLD=WORK/'domv-audited-20261012'
ROOT=WORK/'krvp-scale-ddp-20261014'
SOURCE=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100')
AUDIT=WORK/'setup-existing-data-domv-20261012'
sys.path.insert(0,str(SOURCE))
from utils.dom_experiment import VARIANTS,atomic_json,digest,file_digest,source_digest

def checked(path):
    value=json.loads(path.read_text())
    assert digest({k:v for k,v in value.items() if k!='fingerprint'})==value['fingerprint']
    return value

def main():
    assert not ROOT.exists(),'Family exists; inspect it before retrying freeze'
    old=checked(OLD/'manifest.json')
    assert old['source_sha256']==source_digest(SOURCE)=='f991015a229476400e0fc772177af33759d1bc3f23901381e2f83c7faaeab921'
    assert VARIANTS['krvp100']==(True,True,'causal_rules')
    assert old['split_sha256']=='bbeca518e7f6ee85c39e5863a4d6745c36de53f13b8831595ca36c304ff9950c'
    assert old['train_frames']==22370201 and old['counts']=={'train':197198,'validation':2015,'test':2011}
    for name,expected in old['artifacts'].items():assert file_digest(OLD/name)==expected,name
    weights=OLD/'initialization/model.safetensors'
    assert weights.is_file() and file_digest(weights)==old['weights_sha256']
    report=json.loads((AUDIT/'verified.json').read_text())
    receipt=json.loads((AUDIT/'combined-verification-receipt.json').read_text())
    assert file_digest(AUDIT/'verified.json')==receipt['sha256']
    assert report['manifest']==old['fingerprint'] and report['state']=='complete' and not report['errors']
    assert report['completed_files']==report['total_files']==804900
    ROOT.mkdir();(ROOT/'provenance').mkdir();(ROOT/'logs').mkdir()
    for name in old['artifacts']:
        path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to((OLD/name).resolve())
    (ROOT/'phase_cache').symlink_to((OLD/'phase_cache').resolve(),target_is_directory=True)
    (ROOT/'initialization/model.safetensors').symlink_to(weights.resolve())
    extra=['native_benchmark.py','multi_entry.py','run_nodes.sh','node_torchrun.sh','stop_ranks.py',
        'scale_check.py','approve_runtime.py','prepare_production.py','combined.py','production.sbatch']
    for name in extra:(ROOT/name).write_bytes((HERE/name).read_bytes())
    manifest=dict(old);manifest.pop('fingerprint')
    # The parent baseline metadata is historical, not the identity of this run.
    manifest['parent_baseline_metadata']=manifest.pop('deltaai_distributed_baseline')
    manifest.update(initialization=str(ROOT/'initialization'),runtime_adaptation_parent=old['fingerprint'],
        run_variant='krvp100',experiment={'variant':'krvp100','residual':True,'velocity':True,
        'phase_gate':'causal_rules','formulation':'coordinate','world_size':16,'node_count':4,
        'initialization':'Original pinned pretrained model; separate fresh run, not a DOM checkpoint',
        'global_effective_batch':352,'runtime_microbatch':22,'runtime_workers':4,
        'readiness':'One actual16-rank smoke and200-update full-data check within15minutes; then same-allocation training',
        'audit':'Reuse exact804900-file audit; dataset/split/causal phase cache/normalization unchanged',
        'simulator_evaluation':'Disabled on DeltaAI; transfer trained checkpoints for evaluation'})
    manifest['distributed_study']=dict(manifest['distributed_study'],supported_world_sizes=[16],
        preferred_world_size=16,first_variant='krvp100')
    manifest['artifacts']=dict(old['artifacts'],**{name:file_digest(ROOT/name) for name in extra})
    manifest['fingerprint']=digest(manifest);atomic_json(ROOT/'manifest.json',manifest)
    recipe=checked(OLD/'recipe.json');recipe.pop('fingerprint');recipe.pop('runtime_fingerprint',None)
    recipe.update(manifest=manifest['fingerprint'],supported_world_sizes=[16],variant='krvp100')
    recipe['fingerprint']=digest(recipe);atomic_json(ROOT/'recipe.json',recipe)
    derived=dict(report);derived.update(manifest=manifest['fingerprint'],verification_reused=True,
        original_verified_manifest=old['fingerprint'],original_verified_receipt_sha256=receipt['sha256'])
    atomic_json(HERE/'verified.json',derived)
    atomic_json(ROOT/'provenance/verification-reuse.json',{'original_receipt_sha256':receipt['sha256'],
        'derived_receipt_sha256':file_digest(HERE/'verified.json'),'original_report':report,
        'original_manifest':old['fingerprint'],'same_dataset_root':old['dataset_root'],
        'all_parent_artifact_hashes_retained':True,'phase_cache':str((ROOT/'phase_cache').resolve()),
        'read_only_cache':True,'new_scan_performed':False})
    atomic_json(ROOT/'provenance/parent-manifest.json',old)
    deployment={'family':str(ROOT),'source':str(SOURCE),'variant':'krvp100','world_size':16,'node_count':4,
        'source_sha256':manifest['source_sha256'],'manifest':manifest['fingerprint'],
        'recipe':recipe['fingerprint'],'split_sha256':manifest['split_sha256'],
        'verified_files_reused':804900,'weights_sha256':old['weights_sha256'],
        'parent_manifest':old['fingerprint'],'fresh_pretrained_initialization':True,
        'model_code':'unchanged shared frozen model; named variant enables residual, velocity and causal gate'}
    atomic_json(HERE/'deployment.json',deployment)
    atomic_json(HERE/'status.json',dict(deployment,stage='ready_to_submit',training_job=None,
        dependency=None,verification_complete=True,verification_reused=True,verified_files=804900,
        readiness_complete=False,readiness_total_limit_minutes=15,training_started=False))
    print(json.dumps(deployment),flush=True)

if __name__=='__main__':main()
