#!/usr/bin/env python3
"""Qualify both heavy policies on excluded, identical serial/parallel cases."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import run_low_clutter_table1 as original
import run_parallel_low_clutter_table1 as production
from monitor_table1_latency import log_samples,coverage
from capture_table1_runtime import stats

OUT=original.ROOT/'runtime_health'
MODELS=['vlash_kr','pi05_baseline']


def lane(catalog,name,folder,index,listing,cancel,shutdown=None):
 folder.mkdir(parents=True,exist_ok=False)
 server,client=production.commands(catalog,name,folder,listing,2,index)
 env=dict(os.environ,DOM_OFFICIAL_SOURCE=catalog['official_source'],DOM_TRIAL_RESULTS=str(folder/'trial_results.jsonl'),
  DOM_EXPECTED_SIM='4.5.0',DOM_EXPECTED_LAB='2.2.1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
  OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 original.write(folder/'commands.json',{'server':server,'client':client,'scored':False})
 processes=[];started=time.monotonic()
 try:
  with (folder/'server.log').open('w') as slog,(folder/'client.log').open('w') as clog:
   processes.append(subprocess.Popen(server,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=slog,stderr=subprocess.STDOUT,start_new_session=True))
   processes.append(subprocess.Popen(client,env=dict(env,CUDA_VISIBLE_DEVICES='1'),stdout=clog,stderr=subprocess.STDOUT,start_new_session=True))
   while processes[1].poll() is None:
    if cancel.is_set() or (shutdown is not None and shutdown.is_set()):raise InterruptedError('Qualification cancelled')
    if processes[0].poll() is not None:raise RuntimeError('Qualification simulator exited: '+str(folder))
    if time.monotonic()-started>1200:raise TimeoutError('Qualification exceeded 20 minutes: '+str(folder))
    time.sleep(1)
   if processes[1].returncode:raise RuntimeError('Qualification client failed: '+str(folder))
  verified=original.verify(folder,original.ROOT/'results'/name,listing,2)
  samples=log_samples(folder/'client.log');stale=[r['stale'] for r in samples if r['stale'] is not None]
  records=original.runtime.read_records(folder)
  result={**stats([r['ms'] for r in samples]),'action_coverage':coverage(folder/'client.log'),
   'expired_chunk_fraction':sum(x>=20 for x in stale)/len(stale) if stale else 0,
   'outcomes':[r['success'] for r in records],'verified':verified,'source_folder':str(folder),
   'wall_seconds':time.monotonic()-started,'scored':False}
  original.write(folder/'performance.json',result);return result
 except BaseException as error:
  cancel.set();original.write(folder/'qualification_failure.json',{'error':repr(error),'scored':False});raise
 finally:
  for process in reversed(processes):original.stop_group(process)


def qualify(catalog,shutdown=None):
 receipt=OUT/'heavy_qualification.json'
 if receipt.exists():return original.read(receipt)
 root=OUT/'heavy_qualification'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 root.mkdir(parents=True)
 listing=root/'cases.txt'
 paths=[(original.ROOT/'test_lists'/f'{d}.txt').read_text().splitlines()[0] for d in ['CR','LS','SR']]
 listing.write_text('\n'.join(paths)+'\n')
 result={'created_utc':datetime.now(timezone.utc).isoformat(),'models':MODELS,'cases':paths,'repeats_per_case':2,
  'qualification_source_sha256':original.sha(__file__),
  'protocol_sha256':catalog['protocol_sha256'],'all_trials_excluded_from_table1':True,
  'thresholds':{'p50_ratio_max':1.10,'p95_ratio_max':1.15,'coverage_drop_max':.02},
  'serial':{},'parallel':{},'mode':'hold_heavy',
  'limitation':'Three configurations with two repeats each cannot establish equal success rates.'}
 try:
  for name in MODELS:
   if shutdown is not None and shutdown.is_set():raise InterruptedError('Controller stopped')
   original.write(OUT/'qualification_progress.json',{'state':'serial_reference','model':name,'scored':False})
   try:
    result['serial'][name]=lane(catalog,name,root/'serial'/name,0,listing,threading.Event(),shutdown)
    reference=result['serial'][name]
    if reference['expired_chunk_fraction']>.05 or reference['action_coverage']<.85:
     raise RuntimeError('Serial reference has excessive expired chunks or low action coverage; investigate intrinsic inference speed before scoring.')
    original.write(OUT/'references'/f'{name}.json',{**reference,
     'kind':'isolated_same_model_three_config_two_repeat_streaming_reference'})
   except Exception as error:
    if shutdown is not None and shutdown.is_set():raise
    result.setdefault('serial_errors',{})[name]=repr(error)
  if result.get('serial_errors'):
   result['held_models']=list(result['serial_errors'])
   result['mode']='hold_heavy' if len(result['held_models'])==len(MODELS) else 'serial'
   original.write(receipt,result)
   original.write(OUT/'qualification_progress.json',{'state':'complete','decision':result['mode'],'held_models':result['held_models'],'scored':False})
   return result
  try:
   original.write(OUT/'qualification_progress.json',{'state':'parallel_comparison','models':MODELS,'scored':False})
   cancel=threading.Event()
   with ThreadPoolExecutor(max_workers=2) as pool:
    jobs={name:pool.submit(lane,catalog,name,root/'parallel'/name,index,listing,cancel,shutdown) for index,name in enumerate(MODELS)}
    result['parallel']={name:future.result() for name,future in jobs.items()}
   checks={}
   for name in MODELS:
    serial=result['serial'][name];parallel=result['parallel'][name]
    checks[name]={'p50_ratio':parallel['p50_ms']/serial['p50_ms'],'p95_ratio':parallel['p95_ms']/serial['p95_ms'],
     'action_coverage_drop':serial['action_coverage']-parallel['action_coverage'],
     'outcome_pattern_equal':parallel['outcomes']==serial['outcomes'],
     'expired_fraction_increase':parallel['expired_chunk_fraction']-serial['expired_chunk_fraction']}
   result['checks']=checks
   good=all(result['serial'][name]['samples']>=30 and result['parallel'][name]['samples']>=30 for name in MODELS) and all(c['p50_ratio']<=1.10 and c['p95_ratio']<=1.15 and c['action_coverage_drop']<=.02
     and c['expired_fraction_increase']<=.02 and c['outcome_pattern_equal'] for c in checks.values())
   result['mode']='parallel_shared' if good else 'serial'
  except Exception as error:
   if shutdown is not None and shutdown.is_set():raise
   result.update(mode='serial',parallel_error=repr(error))
 except Exception as error:
  if shutdown is not None and shutdown.is_set():raise
  result.update(mode='hold_heavy',serial_error=repr(error))
 original.write(receipt,result)
 original.write(OUT/'qualification_progress.json',{'state':'complete','decision':result['mode'],'scored':False})
 return result
