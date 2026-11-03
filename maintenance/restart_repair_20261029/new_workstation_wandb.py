import wandb,json
from datetime import datetime,timezone
try:
 r=wandb.Api(timeout=30).run('aparame-clemson-university/Kinematic-Residual-VLA/664747c0')
 print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'id':r.id,'name':r.name,'state':r.state,'config':dict(r.config),'summary':dict(r.summary),'url':r.url,'created_at':r.created_at},default=str))
except Exception as e:print(json.dumps({'error':type(e).__name__}))
