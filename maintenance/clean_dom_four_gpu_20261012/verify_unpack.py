import hashlib,json,sys,tarfile,time
from pathlib import Path
root=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/dom-audited-20261012')
archive=root/'dom-clean.tar';expected=(root/'dom-clean.sha256').read_text().strip();h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()==expected,'SCP archive checksum mismatch'
assert not (root/'dataset').exists() and not (root/'family').exists(),'Refusing to overwrite an existing extraction'
with tarfile.open(archive,'r') as tf:
 for member in tf:
  assert member.isfile() and member.name.startswith(('dataset/','family/')) and '..' not in Path(member.name).parts, member.name
  tf.extract(member,root)
source=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100');sys.path.insert(0,str(source))
from utils.dom_experiment import atomic_json,digest,file_digest,source_digest
family=root/'family';m=json.loads((family/'manifest.json').read_text());original=dict(m)
for name,v in m['artifacts'].items():assert file_digest(family/name)==v,name
assert file_digest(family/'initialization/model.safetensors')==m['weights_sha256']
m.pop('fingerprint');m.update(dataset_root=str(root/'dataset'),initialization=str(family/'initialization'),source_sha256=source_digest(source));m['fingerprint']=digest(m);atomic_json(family/'manifest.json',m)
r=json.loads((family/'recipe.json').read_text());r.pop('fingerprint');r['manifest']=m['fingerprint'];r['fingerprint']=digest(r);atomic_json(family/'recipe.json',r)
atomic_json(root/'verified.json',{'state':'complete','archive_sha256':expected,'verified_unix':time.time(),'episodes':m['episodes'],'frames':m['frames'],'split_sha256':m['split_sha256'],'manifest':m['fingerprint'],'source_sha256':m['source_sha256'],'source':str(source),'family':str(family),'palmetto_manifest':original['fingerprint'],'method':'source files individually hash-checked during packing; complete SCP archive SHA256 verified before extraction; family artifact hashes checked after extraction'})
print((root/'verified.json').read_text(),flush=True)
