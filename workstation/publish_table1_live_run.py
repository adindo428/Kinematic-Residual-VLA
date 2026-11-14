#!/usr/bin/env python3
"""Publish public aggregate Table 1 results to a dedicated W&B evaluation run."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

DIMS=['CR','DA','LS','VU','SR','MP','VG','MG','DR']
ENTITY='aparame-clemson-university';PROJECT='Kinematic-Residual-VLA'


def read(path):return json.loads(path.read_text())
def write(path,value):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def snapshot(root,service_active):
 catalog=read(root/'catalog.json');status=read(root/'status.json');columns=['Method',*DIMS,'Success (%)','Time (s)']
 table=[];numeric={};verified_count=0
 active=service_active and status.get('state')=='evaluating'
 active_rows=status.get('active_evaluations',[status]) if active else []
 for name in catalog['order']:
  folder=root/'results'/name;values={}
  for i,dimension in enumerate(DIMS):
   path=folder/'full'/f'{i:02d}_{dimension}'/'verification.json'
   if path.exists():
    verified=read(path)
    if verified.get('verified') and verified.get('matched_prompts_verified') and verified.get('trajectory_metrics_verified'):
     values[dimension]=verified['success_percent'];verified_count+=verified['trials']
  path=folder/'summary.json'
  if path.exists():
   summary=read(path)
   if summary.get('complete'):values.update(summary['observed'])
  cells={key:f'{values[key]:.2f}' if key in values else 'pending' for key in [*DIMS,'Success','Time']}
  label=catalog['models'][name]['label']
  for row in active_rows:
   if row.get('model')!=name:continue
   dimension=row.get('dimension')
   if row.get('phase')=='full' and dimension in DIMS and dimension not in values:cells[dimension]='running'
   elif row.get('phase')=='pilot':label+=' (running pilot)'
  table.append([label,*[cells[key] for key in [*DIMS,'Success','Time']]])
  for key,value in values.items():numeric[f'results/{name}/{key}']=value
 public_lanes=[{'method':catalog['models'].get(row.get('model'),{}).get('label','none'),
  'phase':row.get('phase','none'),'condition':row.get('dimension') or 'none',
  'completed':row.get('completed_records_in_phase',0),'target':row.get('phase_trial_target',0)} for row in active_rows]
 current='; '.join(row['method'] for row in public_lanes) or catalog['models'].get(status.get('model'),{}).get('label','none')
 public_status={'state':status.get('state','unknown') if service_active else 'inactive',
  'current_method':current,'phase':'parallel' if len(public_lanes)>1 else status.get('phase','none'),
  'condition':'; '.join(row['condition'] for row in public_lanes) or status.get('dimension') or 'none',
  'completed_in_current_phase':sum(row['completed'] for row in public_lanes),
  'current_phase_target':sum(row['target'] for row in public_lanes),
  'active_evaluations':public_lanes,'concurrency':status.get('concurrency',1),
  'fully_verified_scored_trials':verified_count,'target_trials_per_checkpoint':1800,'queued_checkpoints':len(catalog['order'])}
 return {'columns':columns,'data':table,'numeric':numeric,'status':public_status}


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
 parser.add_argument('--service',required=True);parser.add_argument('--preview-only',action='store_true');args=parser.parse_args()
 root=args.root
 def active():return subprocess.run(['systemctl','--user','is-active',args.service],capture_output=True,text=True).stdout.strip() in ['active','activating','reloading']
 if args.preview_only:
  write(root/'wandb_public_payload_preview.json',snapshot(root,True));return
 import wandb
 record_path=root/'wandb_live_run.json';existing=record_path.exists()
 if existing:record=read(record_path)
 else:
  record={'entity':ENTITY,'project':PROJECT,'id':uuid.uuid4().hex[:8],'name':'Table1-low-clutter-90cases-20repeats-20261102'}
  record['url']=f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{record["id"]}'
  write(record_path,record)
 settings=wandb.Settings(disable_git=True,disable_code=True,save_code=False,disable_job_creation=True,
  x_disable_meta=True,x_disable_stats=True,x_disable_machine_info=True,console='off')
 run=wandb.init(entity=ENTITY,project=PROJECT,id=record['id'],resume='allow',name=record['name'],
  job_type='evaluation',group='Table1-low-clutter-90x20',tags=['evaluation','table1','low-clutter'],mode='online',settings=settings,
  config={'benchmark':'Low-clutter full Table 1 variant','configurations':90,'repeats_per_configuration':20,
   'trials_per_checkpoint':1800,'task_object_cap':2,'containers_excluded_from_cap':True,
   'original_physical_cases':72,'capped_cases':18,'conditions':DIMS,
   'scope':'All original conditions retained; not an unchanged official replication or an IID-only suite.',
   'comparison_note':'DOM/KRV also differs in neural velocity conditioning. SmolVLA baseline/KR uses the same neural observations.'})
 record.update(state='publishing',updated_utc=datetime.now(timezone.utc).isoformat());write(record_path,record)
 previous_table=None
 try:
  while True:
   running=active();value=snapshot(root,running)
   table_signature=hashlib.sha256(json.dumps(value['data']).encode()).hexdigest()
   metrics={**value['numeric'],'progress/completed_in_current_phase':value['status']['completed_in_current_phase'],
    'progress/fully_verified_scored_trials':value['status']['fully_verified_scored_trials']}
   if table_signature!=previous_table:
    metrics['Table1/results']=wandb.Table(columns=value['columns'],data=value['data'])
    previous_table=table_signature
   run.log(metrics)
   for key,v in value['status'].items():run.summary['evaluation/'+key]=v
   record.update(state='publishing' if running else 'finished_publishing',updated_utc=datetime.now(timezone.utc).isoformat(),
    current_method=value['status']['current_method'],current_condition=value['status']['condition'],
    payload_scope='Aggregate numeric results and running/pending display only; no checkpoint, scene, training provenance or raw trial uploads.')
   write(record_path,record)
   if not running or value['status']['state']=='complete':break
   time.sleep(60)
 finally:
  run.finish()


if __name__=='__main__':main()
