"""Read-only scheduler and metadata evidence; never loads/copies model tensors."""
import json,subprocess,hashlib
from pathlib import Path
from datetime import datetime,timezone
root=Path('/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020');control=root/'control'
def read(path):
 if not path.exists():return None
 try:return json.loads(path.read_text())
 except ValueError:return {'invalid_json':str(path)}
def command(args):
 p=subprocess.run(args,text=True,capture_output=True,timeout=45);return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def metrics(path):
 if not path.exists():return []
 out=[]
 for line in path.read_text(errors='replace').splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if isinstance(r,dict) and 'Train/optimizer_updates' in r:out.append(r)
 return out[-20:]
c=read(control/'config.json');s=read(control/'submissions.json');result={'checked_at':datetime.now(timezone.utc).isoformat(),'submissions':s,'source_manifest':read(root/'source-manifest.json'),'config':c,'profiles':{},'production':{},'parents':{},'errors':{},'handoff':{}}
ids=[r['job_id'] for kind in ['profiles','production'] for r in s[kind].values()]+[r['parent_job_id'] for r in c['runs'].values()]+c['supersedes_pending_h200']
result['squeue']=command(['squeue','-j',','.join(ids),'-h','-o','%i|%T|%R|%N'])
result['sacct']=command(['sacct','-X','-j',','.join(ids),'-n','-P','--format=JobIDRaw,JobName,State,ExitCode,Elapsed,AllocTRES,ReqTRES'])
for name,spec in c['runs'].items():
 result['profiles'][name]=read(root/'profiles'/(name+'.json'))
 for category,parent,job in [('production',root,s['production'][name]['job_id']),('parents',Path(spec['parent_root']),spec['parent_job_id'])]:
  out=parent/'runs'/name
  row={n:read(out/(n+'.json')) for n in ['provenance','latest','completed','resume-migration','resume-inference','stopped','wandb','wandb-error']}
  row['latest_checkpoint_exists']=(out/'latest.pt').exists();row['metrics']=metrics(parent/'logs'/(name+'-train-'+job+'.out'))
  for ext in ['out','err']:
   p=parent/'logs'/(name+'-train-'+job+'.'+ext);row['log_'+ext+'_tail']='\n'.join(p.read_text(errors='replace').splitlines()[-20:]) if p.exists() else None
  result[category][name]=row
for p in (control/'errors').glob('*.json') if (control/'errors').exists() else []:result['errors'][p.name]=read(p)
for p in (control/'handoff').rglob('*.json') if (control/'handoff').exists() else []:result['handoff'][str(p.relative_to(control/'handoff'))]=read(p)
for name,row in s['profiles'].items():
 result.setdefault('profile_logs',{})[name]={}
 for ext in ['out','err']:
  p=root/'logs'/(name+'-profile-'+row['job_id']+'.'+ext);result['profile_logs'][name][ext]='\n'.join(p.read_text(errors='replace').splitlines()[-25:]) if p.exists() else None
print(json.dumps(result))
