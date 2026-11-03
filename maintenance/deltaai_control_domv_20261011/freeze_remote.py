import json
from pathlib import Path
import sys
WORK=Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
OLD=WORK/'domv-lean-ddp-20261010'
ROOT=WORK/'domv-control-ddp-20261011'
SOURCE=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261011-control-gloo/dom100')
sys.path.insert(0,str(SOURCE))
from utils.dom_experiment import atomic_json,digest,file_digest,source_digest
assert not ROOT.exists()
ROOT.mkdir()
original=json.loads((OLD/'manifest.json').read_text())
check=dict(original);assert digest({k:v for k,v in check.items() if k!='fingerprint'})==check['fingerprint']
for name,value in original['artifacts'].items():
    assert file_digest(OLD/name)==value,name
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.symlink_to(OLD/name)
for name in ('phase_cache','initialization'):
    path=ROOT/name
    if not path.exists():path.symlink_to((OLD/name).resolve(),target_is_directory=True)
# initialization/config.json is an artifact; its parent must also expose weights.
weights=ROOT/'initialization/model.safetensors'
if not weights.exists():weights.symlink_to((OLD/'initialization/model.safetensors').resolve())
manifest=dict(original);manifest.pop('fingerprint')
manifest['initialization']=str(ROOT/'initialization')
manifest['source_sha256']=source_digest(SOURCE)
manifest['runtime_adaptation_parent']=original['fingerprint']
manifest['fingerprint']=digest(manifest);atomic_json(ROOT/'manifest.json',manifest)
recipe=json.loads((OLD/'recipe.json').read_text());recipe.pop('fingerprint')
recipe['manifest']=manifest['fingerprint'];recipe['fingerprint']=digest(recipe)
atomic_json(ROOT/'recipe.json',recipe)
(ROOT/'provenance').mkdir();atomic_json(ROOT/'provenance/parent-manifest.json',original)
(ROOT/'logs').mkdir()
print(json.dumps({'family':str(ROOT),'source':str(SOURCE),'manifest':manifest['fingerprint'],
    'source_sha256':manifest['source_sha256'],'recipe':recipe['fingerprint']}))
