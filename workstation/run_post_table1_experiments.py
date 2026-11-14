#!/usr/bin/env python3
"""Wait for completed scored suites, then run bounded compute-only follow-ups."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_low_clutter_table1 as original

ROOT=original.ROOT
OUT=ROOT/'post_table1'
SERVICE='krvla-low-clutter-full-table1-20261102.service'


def ready(catalog):
 if original.read(ROOT/'status.json')['state']!='complete':return False
 for name in catalog['order']:
  path=ROOT/'results'/name/'summary.json'
  if not path.exists():return False
  summary=original.read(path)
  if not summary.get('complete') or summary.get('verified_trials')!=1800:return False
 active=subprocess.run(['systemctl','--user','is-active',SERVICE],capture_output=True,text=True).stdout.strip()
 if active in ['active','activating','deactivating']:return False
 lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
 inference_gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 return inference_gpu not in [line.strip() for line in lines]


def bootstrap():
 import numpy as np
 results=[]
 for baseline,kr in [('dom100_best','krv100_best'),('dom100_latest','krv100_latest'),('smolvla_baseline','smolvla_kr')]:
  path=ROOT/'paired_analysis'/(baseline+'__vs__'+kr)/'all_paired_trials.csv'
  with path.open() as stream:rows=list(csv.DictReader(stream))
  assert len(rows)==1800
  by_config={}
  for row in rows:
   by_config.setdefault(row['scenario'],[]).append(int(row['kr_success']=='True')-int(row['baseline_success']=='True'))
  assert len(by_config)==90 and all(len(values)==20 for values in by_config.values())
  values=np.asarray([np.mean(by_config[k]) for k in sorted(by_config)])
  generator=np.random.default_rng(42)
  samples=100*values[generator.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
  results.append({'baseline':baseline,'kr':kr,'paired_trials':1800,'configuration_clusters':90,'resamples':10000,
   'seed':42,'success_difference_pp':float(values.mean()*100),'ci95_pp':np.quantile(samples,[.025,.975]).tolist(),
   'interpretation':'Configuration-cluster percentile bootstrap conditional on these checkpoints; not variation across training seeds.'})
 original.write(OUT/'paired_bootstrap.json',{'complete':True,'pairs':results})


def summarize_profiles(catalog):
 models={}
 for name in catalog['order']:
  row=original.read(OUT/'controlled_runtime'/f'{name}.json')
  models[name]={k:v for k,v in row.items() if k not in ['enabled','analytic_disabled']}
  for mode in ['enabled','analytic_disabled']:
   if mode in row:models[name][mode]={k:v for k,v in row[mode].items() if k!='raw_samples'}
 pairs=[]
 for b,k in [('dom100_best','krv100_best'),('dom100_latest','krv100_latest'),('smolvla_baseline','smolvla_kr')]:
  baseline=models[b]['enabled']['p50_ms'];kr=models[k]['enabled']['p50_ms']
  run_deltas=[y['p50_ms']-x['p50_ms'] for x,y in zip(models[b]['enabled']['runs'],models[k]['enabled']['runs'])]
  pairs.append({'baseline':b,'kr':k,'delta_p50_ms':kr-baseline,'delta_p50_percent':100*(kr/baseline-1),
   'paired_run_delta_p50_ms':run_deltas,'parameter_difference':models[k]['total_parameters']-models[b]['total_parameters'],
   'isolates_KR':b=='smolvla_baseline','confound':None if b=='smolvla_baseline' else 'Neural velocity conditioning also changes.'})
 for name,row in models.items():
  if 'analytic_disabled' in row:
   enabled=row['enabled']['p50_ms'];disabled=row['analytic_disabled']['p50_ms']
   row['same_checkpoint_analytic_delta_p50_ms']=enabled-disabled
   row['same_checkpoint_analytic_delta_percent']=100*(enabled/disabled-1)
 original.write(OUT/'controlled_runtime/summary.json',{'complete':True,'models':models,'pairs':pairs,
  'measurement_class':'sequential synchronized synthetic-input compute-only profile; not manipulation evaluation'})


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
 OUT.mkdir(exist_ok=True)
 catalog=original.read(ROOT/'catalog.json')
 original.validate(catalog)
 plan={'state':'waiting_for_all_scored_suites','created_utc':datetime.now(timezone.utc).isoformat(),
  'requires_complete_scored_trials':1800*len(catalog['order']),'main_evaluation_interrupted':False,
  'primary_cohort':catalog.get('primary_order',catalog['order']),'historical_comparisons':catalog.get('historical_order',[]),
  'jobs':['paired configuration bootstrap, 10000 resamples','sequential controlled compute-only profiling of '+str(len(catalog['order']))+' checkpoints'],
  'controlled_profile':{'warmup':20,'samples_per_run':200,'repeats':5,'examples':12,'batch_size':1,'horizon':20,
   'input_kind':'fixed seeded synthetic RGB/state/velocity and three prompts, matching within pairs',
   'hardware':'one otherwise compute-idle RTX A6000, CUDA_VISIBLE_DEVICES=1'},
  'not_scheduled':['new model training','additional closed-loop velocity/OOD trials','visual velocity-estimator deployment'],
  'pending_reasons':'Separate matched ablation weights, speed-distribution membership, or a causal estimator are not yet specified.',
  'worker_sha256':original.sha(Path(__file__).with_name('profile_table1_controlled.py'))}
 if args.prepare_only or not (OUT/'plan.json').exists():original.write(OUT/'plan.json',plan)
 if args.prepare_only:return
 while not ready(catalog):
  original.write(OUT/'status.json',{'state':'waiting_for_all_scored_suites_and_idle_inference_gpu','updated_utc':datetime.now(timezone.utc).isoformat()})
  time.sleep(30)
 original.validate(catalog)
 assert original.sha(Path(__file__).with_name('profile_table1_controlled.py'))==original.read(OUT/'plan.json')['worker_sha256']
 subprocess.run([sys.executable,str(original.REPO/'workstation/analyze_low_clutter_pairs.py'),'--root',str(ROOT)],check=True)
 bootstrap()
 for name in catalog['order']:
  path=OUT/'controlled_runtime'/f'{name}.json'
  if path.exists():continue
  while not ready(catalog):time.sleep(30)
  original.write(OUT/'status.json',{'state':'controlled_profile','model':name,'updated_utc':datetime.now(timezone.utc).isoformat()})
  entry=catalog['models'][name]
  python=str(original.WORK/'envs/train/bin/python') if entry['client_kind']=='native' else entry.get('python',catalog['python'])
  command=[python,'-u',str(original.REPO/'workstation/profile_table1_controlled.py'),
   '--catalog',str(ROOT/'catalog.json'),'--model',name,'--output',str(path)]
  path.parent.mkdir(exist_ok=True)
  with (path.parent/f'{name}.log').open('a') as stream:
   subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=7200)
 summarize_profiles(catalog)
 subprocess.run([sys.executable,str(original.REPO/'workstation/update_paper_table1.py')],check=True)
 original.write(OUT/'status.json',{'state':'complete','updated_utc':datetime.now(timezone.utc).isoformat(),
  'scored_evaluation_results_changed':False})


if __name__=='__main__':main()
