import json
from pathlib import Path
import wandb
api=wandb.Api(timeout=40)
results={}
for name,runid in [('workstation_kr_full','2qh0ijb1'),('workstation_dom_fresh20k','664747c0'),('domv100_parent_history','bwx4b5i1')]:
 run=api.run('aparame-clemson-university/Kinematic-Residual-VLA/'+runid)
 rows=list(run.scan_history(keys=['_step','Validation/h20/action_mse','Stop/completed_epochs'],page_size=100000))
 results[name]={'id':runid,'rows':rows}
 print(name,json.dumps(rows),flush=True)
folder=Path('/scratch/aparame/Dynamic_VLA/main8-replicas-h200-20261021/runs/vlash_kr')
results['vlash_kr']={'validation':[{'file':p.name,**json.loads(p.read_text())} for p in folder.glob('validation-*.json')],'files':[p.name for p in folder.iterdir() if p.is_file()]}
print('VLASHKR',json.dumps(results['vlash_kr']['validation']),flush=True)
Path('/tmp/krvla-validation-history-20261029.json').write_text(json.dumps(results,indent=2)+'\n')
