"""Relocate the exact Palmetto DDP preparation and freeze its paired recipe."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

work=Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
family=work/'domv-ddp-es-20261009'
source=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261009T165212Z/dom100')
sys.path.insert(0,str(source))
from utils.dom_experiment import digest,file_digest,source_digest,atomic_json
assert not family.exists()
family.mkdir()
payload=sys.stdin.buffer.read()
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
    for member in archive.getmembers():
        path=Path(member.name)
        assert member.isfile() and not path.is_absolute() and '..' not in path.parts
    archive.extractall(family)
original=json.loads((family/'manifest.json').read_text())
check=dict(original);expected=check.pop('fingerprint');assert digest(check)==expected
for name,value in original['artifacts'].items():assert file_digest(family/name)==value,name
shared=work/'dom100/initialization'
assert file_digest(shared/'model.safetensors')==original['weights_sha256']
assert file_digest(shared/'config.json')==original['artifacts']['initialization/config.json']
# The archive includes a small identical config; replace its directory by the
# immutable shared initialization to avoid copying model weights.
(family/'initialization/config.json').unlink()
(family/'initialization').rmdir()
(family/'initialization').symlink_to(shared,target_is_directory=True)
parent=work/'dom100-palmetto-h20-r2'
prior=json.loads((parent/'manifest.json').read_text())
for key in ('split_sha256','weights_sha256','dataset_revision','velocity_mean','velocity_std','train_frames','phase_config'):
    assert original[key]==prior[key],key
assert file_digest(parent/'phase_hashes.json')==original['artifacts']['phase_hashes.json']
(family/'phase_cache').symlink_to((parent/'phase_cache').resolve(),target_is_directory=True)
manifest=dict(original);manifest.pop('fingerprint')
manifest['dataset_root']=str(work/'huggingface/lerobot/hzxie/DOM')
manifest['initialization']=str(shared)
manifest['source_sha256']=source_digest(source)
manifest['deltaai_distributed_baseline']={'variant':'domv100','paired_palmetto_variant':'krv100',
    'paired_palmetto_job':16869869,'parent_manifest':original['fingerprint'],
    'parent_source_sha256':original['source_sha256'],'split_targets_and_initialization_unchanged':True,
    'simulator_evaluation':'Separate Palmetto evaluation after checkpoint transfer'}
manifest['fingerprint']=digest(manifest)
atomic_json(family/'manifest.json',manifest)
recipe=json.loads((family/'recipe.json').read_text())
check=dict(recipe);expected=check.pop('fingerprint');assert digest(check)==expected
provenance=family/'provenance';provenance.mkdir()
atomic_json(provenance/'palmetto-manifest.json',original)
atomic_json(provenance/'palmetto-recipe.json',recipe)
recipe.pop('fingerprint')
recipe['manifest']=manifest['fingerprint']
recipe['simulator_evaluation_mode']='disabled_on_deltaai'
recipe['evaluation_plan_sha256']=None
assert recipe['epochs']==100 and recipe['early_stopping']['relative_min_delta']==0.005
recipe['fingerprint']=digest(recipe)
atomic_json(family/'recipe.json',recipe)
report={'family':str(family),'source':str(source),'manifest':manifest['fingerprint'],
    'recipe':recipe,'source_sha256':manifest['source_sha256'],'split_sha256':manifest['split_sha256'],
    'verified_metadata_artifacts':len(original['artifacts']),'training_submitted':False}
atomic_json(family/'DEPLOYMENT_REPORT.json',report)
print(json.dumps(report))
