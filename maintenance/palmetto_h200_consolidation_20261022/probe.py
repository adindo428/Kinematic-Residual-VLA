import json,subprocess
from pathlib import Path
base=Path('/scratch/aparame/Dynamic_VLA')
result={}
for family,names in [('main8-replicas-a100-20261021',['pi05_baseline','pi05_kr','vlash_baseline','vlash_kr']),('main8-replicas-h200-20261021',['pi05_baseline','pi05_kr','vlash_baseline','vlash_kr']),('main8-20k-10epochs-20261021-r2',['smolvla_baseline','smolvla_kr','domv100'])]:
 root=base/family;records={}
 for name in names:
  out=root/'runs'/name;record={}
  for filename in ['latest.json','provenance.json','completed.json','stopped.json','wandb.json','subset-migration.json']:
   file=out/filename
   if file.is_file():record[filename]=json.loads(file.read_text())
  profile=root/'profiles'/(name+'.json')
  if profile.is_file():record['profile']=json.loads(profile.read_text())
  record['checkpoint_files']=[{'path':str(p),'bytes':p.stat().st_size} for p in out.glob('*.pt')]
  records[name]=record
 result[family]=records
print(json.dumps(result))
