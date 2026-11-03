"""Read-only scheduler/receipt/checkpoint metadata probe; no mutation or weight transfer."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
root=Path('/scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-opt-r2-20261020')
control=Path('/scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-control-20261020')
parent=Path('/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-pi05-cache-20261020')
def read(p):return json.loads(p.read_text())
state=read(root/'submissions.json');handoff=read(control/'submissions.json') if (control/'submissions.json').exists() else {}
ids=[state['paired_profile']['job_id'],'17719993','17720948','17709592','17709594']
if handoff.get('verification'):ids.append(handoff['verification']['job_id'])
ids.extend(v['job_id'] for v in handoff.get('production',{}).values())
ids=list(dict.fromkeys(ids));q=subprocess.run(['squeue','-j',','.join(ids),'-h','-o','%i|%T|%R'],text=True,capture_output=True);a=subprocess.run(['sacct','-X','-j',','.join(ids),'-n','-P','--format=JobIDRaw,State,Elapsed,ExitCode'],text=True,capture_output=True)
if q.returncode or a.returncode:raise RuntimeError('Scheduler read failed; retain last verified state: '+q.stderr+a.stderr)
files={n:(root/n).read_text() for n in ['submissions.json','preflight.json','source-manifest.json','prepared.json','README.md']}
artifacts={str(p.relative_to(root)):read(p) for folder in ['tuning','profiles'] for p in (root/folder).glob('**/*.json') if p.name in ['attempts.json','selection.json'] or p.parent==root/'profiles'}
if (root/'profile-complete.json').exists():artifacts['profile-complete.json']=read(root/'profile-complete.json')
logs={str(p.relative_to(root)):p.read_text(errors='replace')[-2500:] for p in (root/'logs').glob('*') if p.is_file() and 'profile-pair' in p.name}
def evidence(family,name,job):
 out=family/'runs'/name;path=family/'logs'/f'{name}-train-{job}.out';metrics=[]
 if path.exists():
  for line in path.read_text(errors='replace').splitlines():
   try:m=json.loads(line)
   except ValueError:continue
   if 'Train/samples_per_second' in m:metrics.append(m)
 receipt={'job_id':job,'metrics':metrics[-30:],'log_modified_at':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat() if path.exists() else None}
 for key in ['latest','provenance','wandb','resume-migration','resume-inference','completed','stopped','wandb-error']:
  p=out/(key+'.json');receipt[key]=read(p) if p.exists() else None
 receipt['own_production_checkpoint_exists']=(out/'latest.pt').is_file()
 return receipt
refs={n:evidence(parent,n,j) for n,j in [('pi05_baseline','17719993'),('pi05_kr','17720948')]}
production={n:evidence(root,n,v['job_id']) for n,v in handoff.get('production',{}).items()}
metadata={n:read(control/n) for n in ['submissions.json','verified.json','remote-preflight.json','expired-native-dependency.json','config.json','control-manifest.json'] if (control/n).exists()}
handoff_artifacts={str(p.relative_to(control)):read(p) for p in (control/'handoff').glob('**/*.json') if p.is_file()}
errors={str(p.relative_to(control)):read(p) for p in (control/'errors').glob('*.json')}
verifylogs={str(p.relative_to(control)):p.read_text(errors='replace')[-2000:] for p in (control/'logs').glob('verify-*') if p.is_file()}
print(json.dumps({'checked_at':datetime.now(timezone.utc).isoformat(),'files':files,'queue':q.stdout,'accounting':a.stdout,'artifacts':artifacts,'logs':logs,'a100_references':refs,'h200_production':production,'handoff_control':metadata,'handoff_artifacts':handoff_artifacts,'handoff_errors':errors,'verification_logs':verifylogs}))
