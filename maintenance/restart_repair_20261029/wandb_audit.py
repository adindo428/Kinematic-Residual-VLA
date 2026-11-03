import json,os,netrc
from datetime import datetime,timezone
import wandb
api=wandb.Api(timeout=30);path='aparame-clemson-university/Kinematic-Residual-VLA'
rows=[];seen=set()
def record(run):
 if run.id in seen:return
 seen.add(run.id);rows.append({'id':run.id,'name':run.name,'state':run.state,'url':run.url,'config':dict(run.config),'summary':dict(run.summary),'created_at':run.created_at,'updated_at':getattr(run,'updated_at',None)})
record(api.run(path+'/2qh0ijb1'))
for i,run in enumerate(api.runs(path,order='-created_at',per_page=100)):
 if i>=100:break
 config=dict(run.config)
 if 'krv' in run.name.lower() or config.get('variant')=='krv100' or config.get('recipe',{}).get('name')=='krv100':record(run)
print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'runs':rows},default=str))
