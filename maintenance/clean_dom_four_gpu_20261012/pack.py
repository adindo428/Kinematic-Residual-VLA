"""Pack only audit-approved files, checking each frozen hash while writing."""
import hashlib,io,json,sys,tarfile,time
from pathlib import Path
family,out=map(Path,sys.argv[1:]);manifest=json.loads((family/'manifest.json').read_text());data=Path(manifest['dataset_root']);info=json.loads((data/'meta/info.json').read_text())
entries={}
for eid,h in json.loads((family/'parquet_hashes.json').read_text()).items():
 rel=info['data_path'].format(episode_chunk=int(eid)//info['chunks_size'],episode_index=int(eid));entries['dataset/'+rel]=(data/rel,h)
for rel,v in json.loads((family/'video_hashes.json').read_text()).items():entries['dataset/'+rel]=(data/rel,v['sha256'])
for p in (data/'meta').rglob('*'):
 if p.is_file():entries['dataset/'+str(p.relative_to(data))]=(p,None)
for rel,h in manifest['artifacts'].items():entries['family/'+rel]=(family/rel,h)
for eid,h in json.loads((family/'phase_hashes.json').read_text()).items():
 rel='phase_cache/%06d.npz'%int(eid);entries['family/'+rel]=(family/rel,h)
for rel in ['manifest.json','recipe.json','evaluation_plan.json','provenance/completed-audit.json','provenance/data-correction-deployment.json']:
 p=family/rel
 if p.is_file():entries['family/'+rel]=(p,None)
entries['family/initialization/model.safetensors']=(family/'initialization/model.safetensors',manifest['weights_sha256'])
progress=out.with_suffix('.status.json');partial=out.with_suffix('.partial')
def status(state,count):
 p=progress.with_suffix('.tmp');p.write_text(json.dumps({'state':state,'files':count,'total':len(entries),'archive':str(out),'updated_unix':time.time()}));p.replace(progress)
status('packing',0)
with tarfile.open(partial,'w') as archive:
 for i,(name,(p,expected)) in enumerate(entries.items(),1):
  raw=p.read_bytes()
  if expected and hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('File changed since audit: '+str(p))
  item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o644;archive.addfile(item,io.BytesIO(raw))
  if i%1000==0:status('packing',i)
partial.rename(out)
h=hashlib.sha256()
with out.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
out.with_suffix('.sha256').write_text(h.hexdigest()+'\n');status('complete',len(entries))
