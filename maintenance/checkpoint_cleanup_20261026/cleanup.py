"""Delete only exact user-authorized superseded checkpoint files, fail closed."""
import hashlib,json,os,subprocess,gc
from pathlib import Path
from datetime import datetime,timezone
H=Path(__file__).resolve().parent;B=Path('/scratch/aparame/Dynamic_VLA')
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def permitted(p):
 relative=p.relative_to(B);family=relative.parts[0]
 return p.name in ('latest.pt','readiness.pt') and ('runs' in relative.parts) and ((family.startswith('cross-vla-krfm-')) or (family in ('main8-h100-optimization-20261022','main8-20k-10epochs-20261021-r2','main8-h200-only-20261022') and p.name=='readiness.pt'))
def active_commands():
 q=subprocess.check_output(['squeue','-u','aparame','-h','-o','%i'],text=True).split()
 result=[]
 for j in q:
  if '_' in j:continue
  v=subprocess.check_output(['scontrol','show','job',j],text=True)
  result.extend(s.strip().split('=',1)[1] for s in v.splitlines() if s.strip().startswith('Command='))
 return result

def main():
 import torch
 plan=json.loads((H/'plan.json').read_text());assert len(plan['candidates'])==41 and plan['bytes']==806490103923
 receipt=H/'deletion-receipt.json';state=json.loads(receipt.read_text()) if receipt.exists() else {'scope':plan['scope'],'plan_sha256':digest(H/'plan.json'),'deleted':[],'started_at':datetime.now(timezone.utc).isoformat()}
 assert state['plan_sha256']==digest(H/'plan.json')
 done={r['path'] for r in state['deleted']};commands=active_commands();verified={}
 # All protected latest checkpoints must be complete and schema-valid before any deletion.
 for row in plan['candidates']:
  p=Path(row['protected_latest'])
  if str(p) in verified:continue
  assert p.is_file() and p.resolve()==p,'Protected production state unavailable'
  c=torch.load(p,map_location='cpu',mmap=True,weights_only=False);m=json.loads(p.with_suffix('.json').read_text());out=p.parent;provenance=json.loads((out/'provenance.json').read_text())
  assert c['identity']==m['identity']==provenance['identity'] and c['progress']==m['progress'],'Protected identity/progress mismatch'
  assert c['model'] and c['optimizer']['state'] and c['optimizer']['param_groups'] and c['scheduler']['last_epoch']==c['progress']['updates'] and len(c['rng'])==2,'Incomplete resume state'
  verified[str(p)]={'progress':c['progress'],'identity':c['identity'],'inode':p.stat().st_ino,'bytes':p.stat().st_size}
  del c;gc.collect()
 state['protected_latest_verified']=verified;atomic(receipt,state)
 for row in plan['candidates']:
  p=Path(row['path']);target=Path(row['protected_latest']);assert permitted(p) and not p.is_symlink() and p!=target and str(p) not in plan['protected_symlink_targets']
  if str(p) in done:assert not p.exists();continue
  assert not any(cmd.startswith(row['family']+'/') for cmd in commands),'A current job uses this historical family'
  stat=p.stat();assert stat.st_ino==row['inode'] and stat.st_size==row['bytes'] and stat.st_mtime==row['mtime'],'Candidate changed'
  assert target.is_file() and p.resolve()!=target.resolve() and (stat.st_dev,stat.st_ino)!=(target.stat().st_dev,target.stat().st_ino),'Protected copy not independent'
  newest=json.loads(target.with_suffix('.json').read_text());assert newest['progress']['updates']>=verified[str(target)]['progress']['updates'],'Protected state regressed'
  oldmeta=row.get('metadata') or {};oldprogress=oldmeta.get('progress') or {}
  if oldprogress:assert newest['progress']['updates']>=oldprogress['updates'],'Candidate is newer than protected latest'
  checksum=digest(p)
  now=p.stat();assert (now.st_ino,now.st_size,now.st_mtime)==(stat.st_ino,stat.st_size,stat.st_mtime),'File changed during hashing'
  # Reconcile active jobs again immediately before removing each exact file.
  assert not any(cmd.startswith(row['family']+'/') for cmd in active_commands()),'Historical family became active'
  p.unlink();assert not p.exists() and target.is_file()
  state['deleted'].append({'path':str(p),'bytes':stat.st_size,'sha256':checksum,'protected_latest':str(target),'at':datetime.now(timezone.utc).isoformat()});state['deleted_bytes']=sum(x['bytes'] for x in state['deleted']);atomic(receipt,state)
  print(json.dumps({'deleted_files':len(state['deleted']),'deleted_bytes':state['deleted_bytes'],'path':str(p)}),flush=True)
 for p in verified:assert Path(p).is_file()
 for link in plan['protected_symlink_targets']:assert Path(link).is_file()
 state['complete']=True;state['completed_at']=datetime.now(timezone.utc).isoformat();atomic(receipt,state)
if __name__=='__main__':main()
