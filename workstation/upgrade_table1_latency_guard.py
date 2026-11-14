#!/usr/bin/env python3
"""Drain both current phases before reloading the adaptive controller."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import run_low_clutter_table1 as original
from render_table1_live_status import render

ROOT=original.ROOT
OUT=ROOT/'runtime_health'
SERVICE='krvla-low-clutter-full-table1-20261102.service'


def prop(field):
 return subprocess.run(['systemctl','--user','show',SERVICE,'--property='+field,'--value'],capture_output=True,text=True).stdout.strip()


def process_state(pid):
 try:
  stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return stat[0]
 except FileNotFoundError:return 'gone'


def child_client(parent,port):
 for path in Path('/proc').glob('[0-9]*/stat'):
  try:
   stat=path.read_text().rsplit(')',1)[1].split()
   if int(stat[1])!=parent:continue
   argv=path.with_name('cmdline').read_bytes().decode().split('\0')
   clients=('inference_krvla.py','inference_fresh20k_checkpoint.py','smolvla_table1_client.py','completed_vla_table1_client.py')
   if not any(arg.endswith(clients) for arg in argv):continue
   for flag in ['--img_port','--port']:
    if flag in argv and argv[argv.index(flag)+1]==str(port):return int(path.parent.name)
  except (OSError,ValueError,IndexError):continue
 raise RuntimeError(f'Cannot identify current policy client for port {port}; do not interrupt')


def drain_heartbeat(targets):
 live=[]
 for target in targets:
  folder=Path(target['folder'])
  if (folder/'verification.json').exists() or process_state(target['client_pid']) in ['Z','gone']:continue
  records=folder/'trial_results.jsonl'
  count=len(records.read_text().splitlines()) if records.exists() else 0
  live.append({'lane':target['lane'],'model':target['model'],'phase':'full','dimension':target['dimension'],
   'state':'evaluating','completed_records_in_phase':count,'phase_trial_target':200,'checkpoint_trial_target':1800})
 primary=live[0] if live else {}
 original.status('evaluating',**{k:v for k,v in primary.items() if k!='state'},
  execution_mode='draining_for_latency_control_upgrade',concurrency=len(live),planned_maintenance=True,
  active_evaluations=live,lanes=live)
 render(ROOT,True)


def launch(unit,script,log):
 restart=[] if unit==SERVICE.removesuffix('.service') else ['--property=Restart=on-failure','--property=RestartSec=15']
 subprocess.run(['systemd-run','--user','--unit='+unit,*restart,
  '--property=WorkingDirectory='+str(original.REPO),'--property=TimeoutStopSec=120',
  '/bin/bash','-lc',script+' >> '+str(log)+' 2>&1'],check=True)


def ensure_sidecar(unit,script,log):
 active=subprocess.run(['systemctl','--user','is-active',unit+'.service'],capture_output=True,text=True).stdout.strip()
 if active not in ['active','activating']:
  subprocess.run(['systemctl','--user','reset-failed',unit+'.service'],capture_output=True)
  loaded=subprocess.run(['systemctl','--user','show',unit+'.service','--property=LoadState','--value'],capture_output=True,text=True).stdout.strip()
  if loaded=='loaded':
   subprocess.run(['systemctl','--user','start',unit+'.service'],check=True)
  else:launch(unit,script,log)


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 state=original.read(ROOT/'status.json')
 if state.get('latency_guard_version')==1:
  original.write(OUT/'upgrade_complete.json',{'state':'already_installed'});return
 catalog=original.read(ROOT/'catalog.json');original.validate(catalog)
 parent=int(prop('MainPID'));targets=[]
 for row in state['active_evaluations']:
  assert row['phase']=='full' and row['state']=='evaluating'
  folder=ROOT/'results'/row['model']/'full'/f"{original.DIMS.index(row['dimension']):02d}_{row['dimension']}"
  targets.append({'lane':row['lane'],'model':row['model'],'dimension':row['dimension'],'folder':str(folder),
   'client_pid':child_client(parent,row['image_port'])})
 assert len(targets)==2
 original.write(OUT/'upgrade_plan.json',{'requested_utc':datetime.now(timezone.utc).isoformat(),
  'state':'waiting_for_first_verified_current_phase','controller_pid':parent,'targets':targets,
  'policy':'No scored trial interrupted. First completed lane waits while the other finishes its current phase.'})
 frozen=False
 def interrupted(signum,frame):raise InterruptedError('Latency-control deployment interrupted')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  while not any((Path(t['folder'])/'verification.json').exists() for t in targets):
   assert int(prop('MainPID'))==parent and prop('ActiveState')=='active','Controller changed; do not interrupt'
   time.sleep(.2)
  os.kill(parent,signal.SIGSTOP);frozen=True
  original.write(OUT/'upgrade_status.json',{'state':'draining_other_current_phase','targets':targets,
   'drain_started_utc':datetime.now(timezone.utc).isoformat(),
   'runtime_note':'Completed lane remains idle while the other finishes its current scored phase; contention may reduce during this documented drain.'})
  # Children remain autonomous; only the controller is frozen so neither lane
  # launches another dimension. The other simulator completes all its trials.
  started=time.monotonic()
  for target in targets:
   folder=Path(target['folder'])
   while process_state(target['client_pid']) not in ['Z','gone']:
    drain_heartbeat(targets)
    if time.monotonic()-started>14400:raise TimeoutError('Draining current phase exceeded four hours')
    time.sleep(2)
   original.verify(folder,ROOT/'results'/target['model'],
    ROOT/'results'/target['model']/'test_lists'/f"{target['dimension']}.txt",20)
  original.write(OUT/'upgrade_verified_boundary.json',{'all_current_phases_verified_200_trials':True,'targets':targets})
  subprocess.run(['systemctl','--user','stop','--no-block',SERVICE],check=True)
  for _ in range(100):
   if prop('ActiveState')!='active':break
   time.sleep(.05)
  os.kill(parent,signal.SIGCONT);frozen=False
  subprocess.run(['systemctl','--user','stop',SERVICE],check=True)
  original.status('paused_for_latency_control_upgrade',verified_targets=targets)
  original.write(OUT/'execution_before_guard_reload.json',original.read(ROOT/'parallel_execution.json'))
  # Transient units disappear after stop. Resume under the same service name
  # so public status publishers continue following the authorized queue.
  ensure_sidecar(SERVICE.removesuffix('.service'),
   'exec /bin/bash workstation/run-parallel-low-clutter-table1.sh',ROOT/'parallel-controller.log')
  time.sleep(3)
  # Metrics/monitor helpers may exit if they observe the brief stop. Restart
  # resource capture to load per-phase adaptive schedule labels in any case.
  subprocess.run(['systemctl','--user','stop','krvla-table1-runtime-metrics-20261104.service'],check=False)
  subprocess.run(['systemctl','--user','stop','krvla-latency-monitor-20261104.service'],check=False)
  prefix='source workstation/env.local.sh; exec "$KRVLA_WORK/envs/eval20k/bin/python" -u '
  ensure_sidecar('krvla-table1-runtime-metrics-20261104',prefix+'workstation/capture_table1_runtime.py',ROOT/'runtime_metrics/collector.log')
  ensure_sidecar('krvla-latency-monitor-20261104',prefix+'workstation/monitor_table1_latency.py',OUT/'monitor.log')
  ensure_sidecar('krvla-table1-live-wandb-20261102','exec /bin/bash workstation/run-table1-live-wandb.sh',ROOT/'live-wandb.log')
  original.write(OUT/'upgrade_complete.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
    'all_current_phase_results_preserved':True,'latency_guard_version':1})
 except BaseException as error:
  if frozen:
   try:os.kill(parent,signal.SIGCONT)
   except ProcessLookupError:pass
  original.write(OUT/'upgrade_failure.json',{'error':repr(error),'controller_unfrozen':True})
  raise


if __name__=='__main__':main()
