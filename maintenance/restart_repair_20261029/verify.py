"""CPU revalidation and allocated admission for unchanged H200 NVL continuations."""
import argparse,json,hashlib,os,subprocess,gc
from pathlib import Path
from checkpoint_contract import validate

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def check(s):
 import torch
 root=Path(s['root']);out=root/'runs'/s['name']
 q=subprocess.check_output(['squeue','-u','aparame','-h','-o','%i|%j|%T'],text=True)
 for line in q.splitlines():
  jid,name,state=line.split('|')
  if jid!=os.environ.get('SLURM_JOB_ID') and state!='PENDING':assert name not in s['conflicting_names'],'Another active trainer for this output'
 assert s['name'] in ('pi05_kr','vlash_baseline')
 for f,h in s['files'].items():assert sha(f)==h,'Frozen file changed: '+f
 recipe=json.load(open(root/'recipes'/(s['name']+'.json')));prepared=json.load(open(root/'prepared-palmetto.json'));approval=json.load(open(root/'profiles'/(s['name']+'.json')))
 assert approval['approved'] and approval['hardware']=='NVIDIA H200 NVL'
 source={p.name:sha(p) for p in root.glob('*.py') if not p.name.startswith('._')};assert source==approval['source']==json.load(open(root/'source-manifest.json'))['files']
 for f,h in prepared['files'].items():assert sha(root/f)==h
 assert sha(root/'source-manifest.json')==prepared['source_sha256']
 split=json.load(open(root/'split.json'));assert len(split['splits']['train'])==20000 and len(set(split['splits']['train']))==20000
 assert sha(root/'split.json')==recipe['subset_sha256']=='98436815f85fbfc78d48e4332e8603684f6bae1620400b181767bab1c5736d88'
 runtime={**approval['runtime'],'autocast_cache_enabled':False,'allocator':'expandable_segments:True'}
 identity=fp({'recipe':recipe,'prepared':prepared,'source':source,'hardware':'NVIDIA H200 NVL','world':2,'torch':torch.__version__,'runtime':runtime});assert identity==approval['identity']==s['identity']
 assert (out/'latest.pt').is_file() and not (out/'latest.pt').is_symlink()
 c=torch.load(out/'latest.pt',map_location='cpu',mmap=True,weights_only=False);meta=json.load(open(out/'latest.json'));p=json.load(open(out/'provenance.json'));done=validate(c,meta,recipe,p)
 assert c['progress']['updates']>=s['saved_updates'],'Saved progress regressed'
 return {'name':s['name'],'root':s['root'],'identity':identity,'progress':c['progress'],'complete':done,'checkpoint':str(out/'latest.pt'),'subset_sha256':recipe['subset_sha256'],'torch':torch.__version__,'approval_sha256':sha(root/'profiles'/(s['name']+'.json'))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--name');p.add_argument('--gpu',action='store_true');a=p.parse_args();H=Path(__file__).resolve().parent;specs=json.load(open(H/'specs.json'));results={}
 for n,s in specs.items():
  if a.name and n!=a.name:continue
  r=check(s)
  if a.gpu:
   import torch
   assert torch.cuda.is_available() and torch.cuda.device_count()==2
   devices=[torch.cuda.get_device_properties(i) for i in range(2)]
   assert all(d.name=='NVIDIA H200 NVL' and d.total_memory>=120*1024**3 for d in devices),'Not the approved actual H200 NVL hardware'
   r['allocated_hardware']=[{'name':d.name,'bytes':d.total_memory} for d in devices]
  results[n]=r;gc.collect()
 receipt=H/('allocated-'+a.name+'-'+os.environ.get('SLURM_JOB_ID','unknown')+'.json' if a.gpu else 'cpu-verified.json');receipt.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results),flush=True)
 if a.gpu and results[a.name]['complete']:return 10
 return 0
if __name__=='__main__':raise SystemExit(main())
