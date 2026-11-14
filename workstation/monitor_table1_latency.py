#!/usr/bin/env python3
"""Passive runtime alarms; success-rate changes alone are never a contention test."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import capture_table1_runtime as capture

ROOT=capture.ROOT
OUT=ROOT/'runtime_health'
DIMS=['CR','DA','LS','VU','SR','MP','VG','MG','DR']


def read(path):return json.loads(path.read_text())
def write(path,value):capture.write(path,value)


def log_samples(path):
 result=[];post=False
 if not path.exists():return result
 with path.open(errors='replace') as stream:
  for line in stream:
   if not line.endswith('\n'):continue
   if capture.TASK.search(line):post=False
   if capture.DONE.search(line):post=True
   native=capture.NATIVE.search(line);adapter=None if native else capture.ADAPTER.search(line)
   stamp=capture.CLOCK.search(line)
   if post or not stamp or (not native and not adapter):continue
   at=datetime.fromisoformat(stamp[0].replace(',','.')).replace(tzinfo=capture.LOCAL).astimezone(timezone.utc)
   result.append({'at':at.timestamp(),'ms':1000*float(native[2] if native else adapter[1]),
     'stale':int(adapter[2]) if adapter else None})
 return result


def coverage(path,limit=20):
 if not path.exists():return None
 rows=[]
 for line in path.read_text(errors='replace').splitlines():
  match=capture.DONE.search(line)
  if match:rows.append((int(match[3]),int(match[4])))
 rows=rows[-limit:]
 return sum(b for a,b in rows)/sum(a for a,b in rows) if rows and sum(a for a,b in rows) else None


def reasons(samples,reference,action_coverage):
 stats=capture.stats([row['ms'] for row in samples]);stale=[row['stale'] for row in samples if row['stale'] is not None]
 stats.update(stale_p50=statistics.median(stale) if stale else None,
   expired_chunk_fraction=sum(x>=20 for x in stale)/len(stale) if stale else None,
   recent_action_coverage=action_coverage)
 if len(samples)<100:return [],stats
 if not reference:return ['no_model_specific_reference'],stats
 flags=[]
 if stats['p95_ms']>reference['p95_ms']*1.25 and stats['p50_ms']>reference['p50_ms']*1.15:
  flags.append('latency_above_model_reference')
 if stats['expired_chunk_fraction'] is not None and stats['expired_chunk_fraction']>.05:
  flags.append('more_than_5_percent_wholly_expired_chunks')
 if action_coverage is not None and reference.get('action_coverage') is not None and action_coverage<reference['action_coverage']-.05:
  flags.append('action_coverage_down_5_percentage_points_task_mix_may_confound')
 return flags,stats


def inspect(state,incidents):
 now=datetime.now(timezone.utc);catalog=read(ROOT/'catalog.json');checks=[]
 lanes=state.get('active_evaluations',[])
 controller=Path(state['controller']) if state.get('execution_mode')=='scene_first_sequential_resident_adapters' and state.get('controller') else None
 if controller and (controller/'execution/status.json').exists():
  resident=read(controller/'execution/status.json');job=resident.get('job')
  if resident.get('stage')=='evaluating' and job:
   lanes=[{'lane':0,'state':'evaluating','model':job['model'],'phase':'full' if job['scored'] else 'pilot',
           'dimension':job['dimension'],'folder':job['folder']}]
 for lane in lanes:
  if lane.get('state') not in ['evaluating','verified']:continue
  name=lane['model'];dim=lane.get('dimension');phase=lane.get('phase')
  if phase not in ['full','pilot']:continue
  folder=Path(lane['folder']) if lane.get('folder') else ROOT/'results'/name/('pilot' if phase=='pilot' else 'full/'+f'{DIMS.index(dim):02d}_{dim}')
  key=name+'/'+str(folder.relative_to(ROOT))
  samples=[row for row in log_samples(folder/'client.log') if row['at']>=now.timestamp()-600]
  reference_path=OUT/'references'/f'{name}.json'
  if controller and not reference_path.exists() and (controller/'execution/resident_profile.json').exists():
   profile=read(controller/'execution/resident_profile.json')
   proof=profile['paired_trials'][-1]['models'][name]['single']
   write(reference_path,{**proof,'kind':'isolated_resident_adapter_qualification_host_to_cpu_action',
     'action_coverage':None,'checkpoint_sha256':profile['worker_receipts'][name]['checkpoint_sha256'],
     'created_utc':now.isoformat()})
  if not reference_path.exists():
   first=[]
   for path in sorted((ROOT/'results'/name/'full').glob('*/client.log')):
    first.extend(log_samples(path))
    if len(first)>=500:break
   if len(first)>=100:
    reference={**capture.stats([r['ms'] for r in first[:500]]),
     'kind':'historical_same_model_live_schedule_drift_reference_not_isolation',
     'action_coverage':None,'created_utc':now.isoformat()}
    write(reference_path,reference)
  reference=read(reference_path) if reference_path.exists() else None
  flags,metrics=reasons(samples,reference,coverage(folder/'client.log'))
  runtime_alert='latency_above_model_reference' in flags or 'more_than_5_percent_wholly_expired_chunks' in flags
  old=incidents.get(key,{})
  if runtime_alert:
   since=old.get('since',now.timestamp());count=old.get('checks',0)+1
   incidents[key]={'since':since,'checks':count}
   sustained=count>=3 and now.timestamp()-since>=120
  else:
   incidents.pop(key,None);sustained=False
  condition='insufficient_samples' if len(samples)<100 else ('alert' if sustained else 'warning' if flags else 'within_reference')
  row={'lane':lane['lane'],'model':name,'phase':phase,'dimension':dim,'state':condition,
    'reference_kind':reference.get('kind') if reference else None,'reference':reference,
    'rolling_window_seconds':600,'flags':flags,'metrics':metrics,'sustained':sustained}
  results=folder/'trial_results.jsonl'
  if results.exists():
   complete=[]
   for line in results.read_text().splitlines():
    try:complete.append(json.loads(line))
    except json.JSONDecodeError:pass
   recent=complete[-20:]
   row['recent_outcomes']={'trials':len(recent),'successes':sum(bool(r['success']) for r in recent),
    'invalid_statuses':sum(r['status'] not in [0,1] for r in recent),
    'interpretation':'Descriptive only; task mix and sampling prevent attributing this success fraction to parallel execution.'}
  if sustained:
   request={'mode':'serial','requested_utc':now.isoformat(),'model':name,'phase':phase,'dimension':dim,
    'reason':flags,'effective_boundary':'Next verified phase boundary; current scored trials are retained and flagged for review.',
    'suspect_interval_start_utc':datetime.fromtimestamp(incidents[key]['since'],timezone.utc).isoformat()}
   write(OUT/'mode_request.json',request)
   if not old.get('recorded'):
    with (OUT/'alerts.jsonl').open('a') as stream:stream.write(json.dumps(request)+'\n')
   incidents[key]['recorded']=True
  checks.append(row)
 result={'updated_utc':now.isoformat(),'evaluation_state':state['state'],'checks':checks,
  'quality_claim':'Runtime/streaming checks cannot establish unchanged manipulation success rates.',
  'heavy_model_gate':'VLASH/pi0.5 require isolated-versus-parallel excluded pilot qualification before scored trials after controller upgrade.',
  'thresholds':{'window_seconds':600,'minimum_chunks':100,'p50_ratio':1.15,'p95_ratio':1.25,
   'whole_chunk_expiry_fraction':.05,'action_coverage_drop_absolute':.05,'sustained_seconds':120},
  'serial_requested':(OUT/'mode_request.json').exists()}
 result['controller_controls_loaded']=state.get('latency_guard_version')==1 or bool(controller)
 resources=ROOT/'runtime_metrics/latest_resources.json'
 if resources.exists():
  resource=read(resources)
  fractions={row[0]:float(row[4])/float(row[5]) for row in resource['gpus'] if len(row)>5 and float(row[5])>0}
  result['gpu_memory_fractions']=fractions
  result['gpu_memory_warning']=any(value>.9 for value in fractions.values())
  result['resource_sample_age_seconds']=(now-datetime.fromisoformat(resource['sampled_utc'])).total_seconds()
 write(OUT/'health.json',result)
 text=('# Resident evaluation runtime health' if controller else '# Parallel evaluation runtime health')+'\n\nUpdated '+now.isoformat()+'.\n\n'
 for row in checks:
  m=row['metrics'];text+=f"- {row['model']} {row['dimension'] or 'pilot'}: {row['state']}; {m.get('samples',0)} chunks; p50 {m.get('p50_ms')} ms, p95 {m.get('p95_ms')} ms; coverage {m.get('recent_action_coverage')}.\n"
 text+='\nModel-specific live references detect drift, not causal parallel overhead. Heavy-model qualification is separate. Success/failure fluctuations alone never trigger checkpoint selection or trial removal.\n'
 (OUT/'health.md').write_text(text)


def main():
 OUT.mkdir(parents=True,exist_ok=True);incidents={}
 while True:
  state=read(ROOT/'status.json');inspect(state,incidents)
  if state['state'] in ['complete','failed','interrupted']:return
  time.sleep(30)


if __name__=='__main__':main()
