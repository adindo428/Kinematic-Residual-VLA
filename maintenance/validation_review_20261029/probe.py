import json,os,netrc,base64,urllib.request
from pathlib import Path
from datetime import datetime,timezone
runs={
 'pi05_baseline':('3dab4a55','/scratch/aparame/Dynamic_VLA/main8-replicas-h200-20261021/runs/pi05_baseline'),
 'pi05_kr':('35cdb92a','/scratch/aparame/Dynamic_VLA/main8-replicas-h200-20261021/runs/pi05_kr'),
 'vlash_baseline':('640fdebc','/scratch/aparame/Dynamic_VLA/h200-microbatch-20261022/vlash_baseline/runs/vlash_baseline'),
 'vlash_kr':('9a9e2f23','/scratch/aparame/Dynamic_VLA/h200-microbatch-20261022/vlash_kr/runs/vlash_kr'),
 'smolvla_baseline':('qabxg0en','/scratch/aparame/Dynamic_VLA/main8-h200-only-20261022/runs/smolvla_baseline'),
 'smolvla_kr':('8npmrzpw','/scratch/aparame/Dynamic_VLA/main8-h200-only-20261022/runs/smolvla_kr'),
 'domv100':('bwx4b5i1','/scratch/aparame/Dynamic_VLA/main8-h200-only-20261022/runs/domv100'),
 'workstation_kr_full':('2qh0ijb1',None),
 'workstation_dom_fresh20k':('664747c0',None),
}
key=os.environ.get('WANDB_API_KEY')
if not key:
 auth=netrc.netrc().authenticators('api.wandb.ai');key=auth[2] if auth else None
if not key:raise RuntimeError('W&B authentication missing')
headers={'Content-Type':'application/json','Authorization':'Basic '+base64.b64encode(('api:'+key).encode()).decode()}
query='''query($entity:String!, $project:String!, $run:String!) { project(name:$project,entityName:$entity) { run(name:$run) {name displayName state summaryMetrics historyKeys} } }'''
report={'checked_at':datetime.now(timezone.utc).isoformat(),'runs':{}}
for name,(run_id,path) in runs.items():
 item={'id':run_id,'path':path}
 request=urllib.request.Request('https://api.wandb.ai/graphql',data=json.dumps({'query':query,'variables':{'entity':'aparame-clemson-university','project':'Kinematic-Residual-VLA','run':run_id}}).encode(),headers=headers)
 try:
  with urllib.request.urlopen(request,timeout=20) as response:data=json.load(response)
  run=data.get('data',{}).get('project',{}).get('run')
  if not run:raise RuntimeError(str(data.get('errors','No run'))[:300])
  summary=json.loads(run['summaryMetrics']);keys=run.get('historyKeys',{})
  if isinstance(keys,str):keys=json.loads(keys)
  item.update(name=run['displayName'],state=run['state'],summary={k:v for k,v in summary.items() if isinstance(v,(str,float,int,bool)) and not k.startswith('_')},history_keys=keys)
 except Exception as e:item['wandb_error']=type(e).__name__
 if path:
  folder=Path(path);item['files']=[p.name for p in folder.iterdir() if p.is_file()]
  item['validation']=[]
  for p in sorted(folder.glob('validation-*.json')):
   data=json.loads(p.read_text());item['validation'].append({'file':p.name,**data})
  item['receipts']={p.name:json.loads(p.read_text()) for p in folder.glob('*.json') if p.name in ('latest.json','completed.json','training_complete.json','resume-verification.json','best.json','wandb.json','provenance.json')}
 report['runs'][name]=item
 print('CHECKED',name,flush=True)
Path('/tmp/krvla-validation-review-20261029.json').write_text(json.dumps(report,indent=2)+'\n')
for name,r in report['runs'].items():
 print(json.dumps({'run':name,**r}),flush=True)
