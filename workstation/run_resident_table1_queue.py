#!/usr/bin/env python3
"""Run the qualified, scene-first resident-checkpoint Table 1 schedule."""
import argparse
from datetime import datetime,timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import run_low_clutter_table1 as low
import run_krv_full_two_suites as krv
import pilot_resident_table1 as resident
import resident_table1_schedule as schedule
from monitor_table1_latency import log_samples,coverage as action_coverage
from capture_table1_runtime import stats

ROOT=schedule.ROOT
STOP=resident.STOP

def stamp():return datetime.now(timezone.utc).isoformat()

def publish(stage,**kw):
    low.write(ROOT/'status.json',{'stage':stage,'updated_utc':stamp(),**kw})
    from render_resident_table1 import render
    render()

def prepare():
    ledger=schedule.freeze()
    request=ROOT/'request.json'
    if not request.exists():
        files=[low.REPO/'workstation'/n for n in ['resident_table1_worker.py','resident_table1_server.py',
               'resident_table1_client.py','resident_table1_schedule.py','pilot_resident_table1.py',
               'run_resident_table1_queue.py','table1_resume_records.py','table1_two_pass.py']]
        low.write(request,{'enabled':True,'user_authorized':True,'models':ledger['models'],
                  'schedule':'scene first; all missing checkpoint blocks for each frozen configuration',
                  'inference':'one active policy; resident inactive GPU weights',
                  'repeats_per_configuration':10,'remaining_ten_repeats_deferred':True,
                  'code_sha256':{str(p):low.sha(p) for p in files},'pilot_required':str(resident.ROOT/'verification.json')})
    return ledger

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    ledger=prepare()
    if args.prepare_only:publish('waiting_for_pilot');return
    signal.signal(signal.SIGTERM,lambda *_:STOP.set());signal.signal(signal.SIGINT,lambda *_:STOP.set())
    with (low.ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (resident.ROOT/'verification.json').exists():
            publish('waiting_for_pilot')
            active=subprocess.run(['systemctl','--user','is-active','krvla-resident-table1-pilot-20261106.service'],capture_output=True,text=True).stdout.strip()
            if active not in ('active','activating'):raise RuntimeError('Resident pilot has not passed; scored scene-first evaluation remains paused')
            if STOP.wait(5):raise InterruptedError('Resident queue paused')
        assert low.read(resident.ROOT/'verification.json')['passed']
        request=low.read(ROOT/'request.json')
        for path,expected in request['code_sha256'].items():assert low.sha(path)==expected,path
        schedule.recover_interrupted(ledger)
        runs=ROOT/'runs';runs.mkdir(exist_ok=True)
        i=1
        while (runs/f'run{i:02d}').exists():i+=1
        run=runs/f'run{i:02d}';plan=schedule.make_plan(ledger,run)
        if not plan['jobs']:publish('first_pass_queue_complete');return
        observe_stop=threading.Event()
        def observe():
            last=-1;changed=time.monotonic()
            try:
                while not observe_stop.wait(5):
                    if STOP.is_set():return
                    for job in plan['jobs']:
                        folder=Path(job['folder'])
                        if job['id'] not in ledger['accepted_jobs'] and (folder/'resident_client_completion.json').exists():schedule.accept(ledger,job)
                    state=low.read(run/'execution/status.json') if (run/'execution/status.json').exists() else {}
                    job=state.get('job');count=0;latency=None
                    if job:
                        folder=Path(job['folder']);records=folder/'trial_results.jsonl'
                        count=len(records.read_text().splitlines()) if records.exists() else 0
                        samples=log_samples(folder/'client.log')[-100:]
                        latency={**stats([r['ms'] for r in samples]),'action_coverage':action_coverage(folder/'client.log')}
                        low.write(folder/'latency_health.json',{'updated_utc':stamp(),'metrics':latency,
                                  'inference':'one active resident worker; other checkpoints idle'})
                    progress=len(ledger['accepted_jobs'])*100+count
                    if progress!=last:last=progress;changed=time.monotonic()
                    publish(state.get('stage','starting'),job=job,block_records=count,latency=latency,run=str(run))
                    if time.monotonic()-changed>1200:raise TimeoutError('No resident evaluation progress for 20 minutes')
            except BaseException as e:
                low.write(run/'observer_failure.json',{'error':repr(e)});STOP.set()
        observer=threading.Thread(target=observe,daemon=True);observer.start()
        try:
            resident.launch(run/'plan.json',run/'execution')
            observe_stop.set();observer.join()
            for job in plan['jobs']:schedule.accept(ledger,job)
            assert all(len(schedule.coverage(ledger,m['key'],d))==100 for m in ledger['models'] for d in low.DIMS)
            publish('first_pass_queue_complete',remaining_repeats_deferred=10)
            low.write(run/'run_status.json',{'state':'complete','accepted_jobs':ledger['accepted_jobs']})
        except BaseException as e:
            observe_stop.set();observer.join()
            state='paused' if STOP.is_set() and not (run/'observer_failure.json').exists() else 'failed'
            publish(state,error=repr(e),run=str(run))
            low.write(run/'run_status.json',{'state':state,'error':repr(e)})
            raise
        finally:STOP.set();observe_stop.set();observer.join(timeout=10)

if __name__=='__main__':main()
