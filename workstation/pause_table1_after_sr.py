#!/usr/bin/env python3
"""Pause the official queue at the verified DOM SR boundary, not mid-trial."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'scratch/dom45/evaluations/low-clutter-full-table1-20261102'
SERVICE='krvla-low-clutter-full-table1-20261102.service'


def read(path):return json.loads(path.read_text())
def write(path,value):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def prop(name):return subprocess.check_output(['systemctl','--user','show',SERVICE,'--property='+name,'--value'],text=True).strip()


def main():
 request={'requested_utc':datetime.now(timezone.utc).isoformat(),'after_model':'dom100_best','after_condition':'SR',
          'reason':'User requests an isolated parallel-evaluation capability benchmark after SR finishes.'}
 write(ROOT/'pause_after_sr_request.json',request)
 verified=ROOT/'results/dom100_best/full/04_SR/verification.json'
 while True:
  if verified.exists():
   try:v=read(verified)
   except json.JSONDecodeError:v={}
   if v.get('verified') and v.get('matched_prompts_verified') and v.get('trajectory_metrics_verified') and v.get('trials')==200:break
  if prop('ActiveState') not in ['active','activating']:raise RuntimeError('Evaluation stopped before verified SR completion')
  time.sleep(.25)
 # Freeze the controller only after durable verification; this prevents it
 # entering MP while systemd processes the stop request. Child rollouts are done.
 pid=int(prop('MainPID'));assert pid>1
 os.kill(pid,signal.SIGSTOP)
 subprocess.run(['systemctl','--user','stop','--no-block',SERVICE],check=True)
 for _ in range(100):
  if prop('ActiveState')!='active':break
  time.sleep(.05)
 try:os.kill(pid,signal.SIGCONT)
 except ProcessLookupError:pass
 subprocess.run(['systemctl','--user','stop',SERVICE],check=True)
 previous=read(ROOT/'status.json');write(ROOT/'status.before_parallel_benchmark.json',previous)
 state={'state':'paused_for_parallel_benchmark','updated_utc':datetime.now(timezone.utc).isoformat(),
  'model':'dom100_best','completed_dimension':'SR','verified_trials_in_dimension':200,
  'reason':request['reason'],'official_results_preserved':True,'benchmark_results_excluded_from_table':True}
 write(ROOT/'status.json',state);write(ROOT/'results/dom100_best/suite_status.json',state)
 # Publish the newly completed SR numbers, even though the frozen controller
 # was stopped before its normal report upload. This never invokes a rollout.
 import run_low_clutter_table1 as runner
 runner.publish(read(ROOT/'catalog.json'),upload=False)
 notice='**Paused after DOM SR for an isolated parallel-evaluation benchmark. Completed results are preserved.**\n\n'
 for path in [ROOT/'table1_comparison.md',REPO/'scratch/dom45/results/krv100-best-epoch003-task-table1-20261025/table1_task_subset.md']:
  text=path.read_text();first,rest=text.split('\n',1);path.write_text(first+'\n\n'+notice+rest.lstrip('\n'))
 write(ROOT/'pause_after_sr_complete.json',{**state,'service_inactive':prop('ActiveState')=='inactive'})
 print('PAUSED_AFTER_VERIFIED_SR',flush=True)


if __name__=='__main__':main()
