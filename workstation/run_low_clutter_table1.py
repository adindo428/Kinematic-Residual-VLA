#!/usr/bin/env python3
"""Run 90 low-clutter Table 1 cases, 20 matched repeats per checkpoint."""
import argparse
import csv
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import run_expert_prompt_matched_eval as runtime
from evaluation_training_policy import require_eligible,comparison_note

REPO=Path(__file__).resolve().parents[1];WORK=REPO/'scratch/dom45'
ROOT=WORK/'evaluations/low-clutter-full-table1-20261102'
PREVIOUS=WORK/'evaluations/expert-prompt-matched20k-20261102'
DIMS=['CR','DA','LS','VU','SR','MP','VG','MG','DR']
ORDER=['dom100_best','krv100_best','dom100_latest','krv100_latest','smolvla_baseline','smolvla_kr','vlash_kr','pi05_baseline']
runtime.ROOT=ROOT
read=runtime.read;write=runtime.write;sha=runtime.sha
verify_trial_folder=runtime.verify_trial_folder
stop_group=runtime.stop_group


def status(state,**details):
 write(ROOT/'status.json',{'state':state,'updated_utc':datetime.now(timezone.utc).isoformat(),**details})


def prepare():
 if (ROOT/'catalog.json').exists():return read(ROOT/'catalog.json')
 assert read(ROOT/'preparation_complete.json')['complete']
 protocol=read(ROOT/'protocol.json');assert len(protocol['tests'])==90 and protocol['repeats']==20 and protocol['scored_trials']==1800
 catalog=read(PREVIOUS/'catalog.json')
 for entry in catalog['models'].values():
  stat=Path(entry['weight_path']).stat()
  assert stat.st_size==entry['weight_bytes'] and stat.st_mtime_ns==entry['weight_mtime_ns'],'Verified checkpoint changed'
 for path,expected in catalog['files_sha256'].items():assert sha(path)==expected,'Prior verified inference source changed: '+path
 catalog.update(order=ORDER,protocol_sha256=sha(ROOT/'protocol.json'),
  checkpoint_verification_receipt=str(PREVIOUS/'catalog.json'),checkpoint_verification_receipt_sha256=sha(PREVIOUS/'catalog.json'),
  paired_comparisons=[['dom100_best','krv100_best'],['dom100_latest','krv100_latest'],['smolvla_baseline','smolvla_kr']],
  comparison_notes={'dom100_vs_krv100':'KRV also adds neural velocity conditioning; this is not an isolated KR-only ablation.',
   'smolvla_pair':'Both use the same neural observations; KR changes the analytic action target/reconstruction.',
   'unpaired_models':'Final VLASH baseline and pi0.5 KR are still awaited; available unpaired counterparts are queued afterward.'})
 for filename in ['run_low_clutter_table1.py','low_clutter_table1_server.py','prepare_low_clutter_table1.py','analyze_low_clutter_pairs.py','run-low-clutter-table1.sh']:
  path=REPO/'workstation'/filename;catalog['files_sha256'][str(path)]=sha(path)
 for name in ORDER:
  entry=catalog['models'][name]
  require_eligible('dom100' if name.startswith('dom100') else 'krv100_fresh' if name.startswith('krv100') else name,
                   entry if entry['client_kind']!='native' else None)
  root=ROOT/'results'/name;shutil.copytree(ROOT/'test_lists',root/'test_lists',dirs_exist_ok=True)
  write(root/'protocol.json',{**protocol,'checkpoint':entry})
  write(root/'suite_status.json',{'state':'queued','scored_trials_target':1800})
 write(ROOT/'catalog.json',catalog);return catalog


def validate(catalog):
 assert sha(ROOT/'protocol.json')==catalog['protocol_sha256']
 protocol=read(ROOT/'protocol.json');assert sha(ROOT/'selection.json')==protocol['selection_sha256']
 for test in protocol['tests']:assert sha(test['path'])==test['config_sha256'],'Frozen scenario changed'
 for path,expected in {**catalog['files_sha256'],**protocol['files_sha256']}.items():assert sha(path)==expected,'Frozen source changed: '+path
 for entry in catalog['models'].values():
  stat=Path(entry['weight_path']).stat();assert stat.st_size==entry['weight_bytes'] and stat.st_mtime_ns==entry['weight_mtime_ns']


def verify(folder,root,listing,repeats):
 verified=verify_trial_folder(folder,root,listing,repeats)
 tests={t['name']:read(t['path']) for t in read(root/'protocol.json')['tests']}
 names=[Path(x).stem for x in listing.read_text().splitlines() if x]
 expected={(name,trial) for name in names for trial in range(1,repeats+1)}
 traces=[json.loads(s) for s in (folder/'prompt_trace.jsonl').read_text().splitlines() if s]
 metrics=[json.loads(s) for s in (folder/'trajectory_metrics.jsonl').read_text().splitlines() if s]
 assert len(traces)==len(expected) and {(r['env_name'],r['trial']) for r in traces}==expected
 assert len(metrics)==len(expected) and {(r['env_name'],r['trial']) for r in metrics}==expected
 for row in traces:
  cfg=tests[row['env_name']];assert row['text']==cfg['evaluation_prompts'][row['trial']-1] and row['seed']==cfg['seed']
 for row in metrics:assert Path(row['trajectory_path']).exists() and row['ee_path_length_m']>=0
 verified.update(matched_prompts_verified=True,trajectory_metrics_verified=True)
 write(folder/'verification.json',verified);return verified


def phase(catalog,name,dimension=None):
 root=ROOT/'results'/name;pilot=dimension is None
 folder=root/'pilot' if pilot else root/'full'/f'{DIMS.index(dimension):02d}_{dimension}'
 listing=root/'test_lists'/('pilot.txt' if pilot else dimension+'.txt');repeats=1 if pilot else 20
 if folder.exists():
  if (folder/'verification.json').exists():return verify(folder,root,listing,repeats)
  raise FileExistsError('Partial output retained; inspect before retry: '+str(folder))
 if not pilot:assert read(root/'pilot/verification.json')['matched_prompts_verified']
 validate(catalog);folder.mkdir(parents=True)
 env=dict(os.environ,DOM_OFFICIAL_SOURCE=catalog['official_source'],DOM_TRIAL_RESULTS=str(folder/'trial_results.jsonl'),
  DOM_EXPECTED_SIM='4.5.0',DOM_EXPECTED_LAB='2.2.1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 client_env=dict(env,CUDA_VISIBLE_DEVICES='1');server_env=dict(env,CUDA_VISIBLE_DEVICES='0')
 client=runtime.client_command(catalog,name,folder)
 if pilot:
  entry=catalog['models'][name]
  if entry['client_kind']=='native':
   probe=[str(WORK/'envs/train/bin/python'),'-u',str(REPO/'workstation/collection_matched_native_probe.py'),
    '--source',entry['source'],'--checkpoint',entry['checkpoint'],'--output',str(folder/'policy_probe.json')]
   if entry['kr']:probe+=['--kr']
  else:probe=[*client,'--probe']
  with (folder/'probe.log').open('w') as log:subprocess.run(probe,env=client_env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
 server=[str(REPO/'workstation/isaaclab-paper45.sh'),'-p',str(REPO/'workstation/low_clutter_table1_server.py'),
  '--headless','--enable_cameras','--device','cuda:0','--num_envs','1','--tolerance','0.07',
  '--scene_dir',str(WORK/'assets/scenes'),'--object_dir',str(WORK/'assets/objects'),
  '--env_cfg',str(listing),'--output_dir',str(folder/'videos'),'-n',str(repeats),'--save',
  '--img_port','27600','--act_port','27601','--kit_args',os.environ['KRVLA_KIT_ARGS']]
 write(folder/'commands.json',{'server':server,'client':client})
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True)
 assert len(gpu.splitlines())==2 and all('RTX A6000' in line for line in gpu.splitlines())
 write(folder/'runtime.json',{'gpu':gpu,'simulation_gpu':0,'inference_gpu':1,'repeats':repeats,'streaming':True})
 processes=[]
 try:
  with (folder/'server.log').open('w') as slog,(folder/'client.log').open('w') as clog:
   processes.append(subprocess.Popen(server,env=server_env,stdout=slog,stderr=subprocess.STDOUT,start_new_session=True))
   processes.append(subprocess.Popen(client,env=client_env,stdout=clog,stderr=subprocess.STDOUT,start_new_session=True))
   changed=time.monotonic();size=-1
   while processes[1].poll() is None:
    if processes[0].poll() is not None:raise RuntimeError('Simulator exited; inspect '+str(folder/'server.log'))
    p=folder/'trial_results.jsonl';new=p.stat().st_size if p.exists() else 0
    if new!=size:
     changed=time.monotonic();size=new
     count=len(p.read_text().splitlines()) if p.exists() else 0
     status('evaluating',model=name,phase='pilot' if pilot else 'full',dimension=dimension,
      completed_records_in_phase=count,phase_trial_target=3 if pilot else 200,checkpoint_trial_target=1800)
    if time.monotonic()-changed>1200:raise RuntimeError('No completed trial for 20 minutes')
    time.sleep(5)
   if processes[1].returncode:raise RuntimeError('Policy exited with error; inspect '+str(folder/'client.log'))
  return verify(folder,root,listing,repeats)
 except BaseException as error:
  write(folder/'failure.json',{'error':str(error),'infrastructure_failure':True});raise
 finally:
  for process in reversed(processes):stop_group(process)


def summarize(catalog,name):
 root=ROOT/'results'/name;records=[];observed={}
 for index,dim in enumerate(DIMS):
  folder=root/'full'/f'{index:02d}_{dim}';v=verify(folder,root,root/'test_lists'/(dim+'.txt'),20)
  assert v['trials']==200;observed[dim]=v['success_percent'];records+=runtime.read_records(folder)
 assert len(records)==1800
 observed.update(Success=100*statistics.mean(r['success'] for r in records),Time=statistics.mean(r['task_time_s'] for r in records))
 write(root/'summary.json',{'complete':True,'verified_trials':1800,'observed':observed,'checkpoint':catalog['models'][name],
  'protocol_sha256':catalog['protocol_sha256']})
 write(root/'suite_status.json',{'state':'complete','verified_trials':1800})


def publish(catalog,upload=False):
 rows=[]
 for name in ORDER:
  root=ROOT/'results'/name;values={}
  for i,d in enumerate(DIMS):
   p=root/'full'/f'{i:02d}_{d}'/'verification.json'
   if p.exists():
    v=read(p)
    if v.get('matched_prompts_verified') and v.get('trajectory_metrics_verified'):values[d]=v['success_percent']
  if (root/'summary.json').exists():values.update(read(root/'summary.json')['observed'])
  rows.append({'Method':catalog['models'][name]['label'],**values})
 fields=['Method',*DIMS,'Success','Time']
 table='| '+' | '.join(fields)+' |\n|'+'|'.join(['---']+['---:']*(len(fields)-1))+'|\n'
 for row in rows:table+='| '+row['Method']+' | '+' | '.join(f'{row[k]:.2f}' if k in row else 'pending' for k in fields[1:])+' |\n'
 text=('# Low-clutter full Table 1 comparison\n\n90 released configurations × 20 repeats = **1,800 scored trials per checkpoint**, plus three excluded pilots. '
  'All nine conditions retained. Task objects are capped at two, excluding containers. Physical configurations are unchanged in 72 cases; 18 are capped. '
  'This is a low-clutter benchmark variant, not an unchanged official replication or an IID-only suite. Same prompts, reset seeds and evaluation settings across methods.\n\n'+table+'\n'+comparison_note()+'\n\n'
  'DOM/KRV also differs in neural velocity conditioning; SmolVLA ± KR retains the same neural inputs. '
  'Paired success transitions and completion-time/trajectory comparisons on jointly successful trials are produced only after both full suites complete. '
  'Success and Time above include all scored trials. Historical results are separate.\n')
 (ROOT/'table1_comparison.md').write_text(text)
 with (ROOT/'table1_comparison.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 write(ROOT/'comparison.json',{'rows':rows,'trials_per_checkpoint':1800})
 ide=WORK/'results/krv100-best-epoch003-task-table1-20261025/table1_task_subset.md'
 if not (ROOT/'previous_display.md').exists():shutil.copy2(ide,ROOT/'previous_display.md')
 ide.write_text(text)
 if upload:
  try:
   import wandb
   dest=ROOT/'reports/low-clutter-table1-90';dest.mkdir(parents=True,exist_ok=True)
   (dest/'results.md').write_text('Low-clutter Table 1 variant: 90 cases, 20 repeats each. Percent success; mean task time in seconds.\n\n'+table)
   shutil.copy2(ROOT/'table1_comparison.csv',dest/'results.csv')
   run=wandb.Api(timeout=30).run('aparame-clemson-university/Kinematic-Residual-VLA/2qh0ijb1')
   for p in dest.iterdir():run.upload_file(str(p),root=str(ROOT))
   write(ROOT/'wandb_upload.json',{'state':'uploaded_numeric_results_only','updated_utc':datetime.now(timezone.utc).isoformat()})
  except Exception as error:write(ROOT/'wandb_upload.json',{'state':'retry_next_phase','error':str(error)})


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'queue.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  catalog=prepare();validate(catalog);publish(catalog)
  if args.prepare_only:status('prepared',order=ORDER,trials_per_checkpoint=1800);return
  for name in ORDER:
   root=ROOT/'results'/name
   if (root/'summary.json').exists():continue
   try:
    status('evaluating',model=name,phase='pilot');write(root/'suite_status.json',{'state':'pilot'})
    phase(catalog,name)
    for dim in DIMS:
     write(root/'suite_status.json',{'state':'full','dimension':dim,'target_trials':1800})
     phase(catalog,name,dim);publish(catalog,upload=True)
    summarize(catalog,name);publish(catalog,upload=True)
    subprocess.run([sys.executable,str(REPO/'workstation/analyze_low_clutter_pairs.py'),'--root',str(ROOT)],check=True)
   except BaseException as error:
    write(root/'suite_status.json',{'state':'failed','error':str(error),'traceback':traceback.format_exc()})
    status('failed',model=name,error=str(error));raise
  status('complete',checkpoints=len(ORDER),scored_trials=1800*len(ORDER));publish(catalog,upload=True)


if __name__=='__main__':main()
